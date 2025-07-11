import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass

def init_bimap_parameter(W):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho,hi,ni,no=W.shape
    for i in range(ho):
        for j in range(hi):
            if ni == no:
                W.data[i, j] = th.eye(ni, no, dtype=W.dtype, device=W.device)
            elif ni<no: #augmentation de dimension pour retourner dans l'espace initial
                v = th.empty(no, no, dtype=W.dtype, device=W.device).uniform_(0., 1.)
                u, s, v_t = th.svd(v.matmul(v.t()))
                W.data[i, j] = v_t.t()[:ni, :]
            else:  #cas no<ni - reduction de dimension pour aller dans l'espace latent
                v=th.empty(ni,ni,dtype=W.dtype,device=W.device).uniform_(0.,1.)
                vv=th.svd(v.matmul(v.t()))[0][:,:no]
                W.data[i,j]=vv

def init_bimap_parameter_identity(W):
    """ initializes to identity a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho,hi,ni,no=W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i,j]=th.eye(ni,no)

class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass

def bimap(X,W): 
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.t().matmul(X).matmul(W)

def bimap_channels(X,W, transpose=False):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size,channels_in,ni,_=X.shape
    channels_out=len(W[0])
    
    no = W[0][0].shape[1 if not transpose else 0]
    
    P = th.zeros((batch_size, channels_out, no, no), dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([
            bimap(X[:, ci, :, :], W[ci][co].T if transpose else W[ci][co])
            for ci in range(channels_in)
        ])
    return P

def modeig_forward(P,op,eig_mode='svd',param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size,channels,n,n=P.shape #batch size,channel depth,dimension
    U,S=th.zeros_like(P,device=P.device),th.zeros(batch_size,channels,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if(eig_mode=='eig'):
                 # j'ai changé ces 3 lignes car deprecated
                L_complex,V_complex = th.linalg.eig(P[i,j])
                U[i,j] = V_complex.real
                S[i,j] = L_complex.real
            elif(eig_mode=='svd'):
                U[i,j],S[i,j],_=th.svd(P[i,j])
    S_fn=op.fn(S,param)
    X=U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2,3))
    return X,U,S,S_fn

def modeig_backward(dx,U,S,S_fn,op,param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv=BatchDiag(op.fn_deriv(S,param))
    SS=S[...,None].repeat(1,1,1,S.shape[-1])
    SS_fn=S_fn[...,None].repeat(1,1,1,S_fn.shape[-1])
    L=(SS_fn-SS_fn.transpose(2,3))/(SS-SS.transpose(2,3))
    L[L==-np.inf]=0; L[L==np.inf]=0; L[th.isnan(L)]=0
    L=L+S_fn_deriv
    dp=L*(U.transpose(2,3).matmul(dx).matmul(U))
    dp=U.matmul(dp).matmul(U.transpose(2,3))
    return dp

class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Log_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Log_op)

class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Re_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Re_op)

class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Exp_op,eig_mode='eig')
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Exp_op)

class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqm_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqm_op)

class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqminv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqminv_op)

class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P,power):
        Power_op._power=power
        X,U,S,S_fn=modeig_forward(P,Power_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Power_op),None

class InvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Inv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Inv_op)

def geodesic(A,B,t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    '''
    M=CongrG(PowerEig.apply(CongrG(B,A,'neg'),t),A,'pos')[0,0]
    return M

def cov_pool(f,reg_mode='mle'):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    bs,n,T=f.shape
    X=f.matmul(f.transpose(-1,-2))/(T-1)
    if(reg_mode=='mle'):
        ret=X
    elif(reg_mode=='add_id'):
        ret=add_id(X,1e-6)
    elif(reg_mode=='adjust_eig'):
        ret=adjust_eig(X,0.75)
    if(len(ret.shape)==3):
        return ret[:,None,:,:]
    return ret

def cov_pool_mu(f,reg_mode):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    alpha=1
    bs,n,T=f.shape
    mu=f.mean(-1,True); f=f-mu
    X=f.matmul(f.transpose(-1,-2))/(T-1)+alpha*mu.matmul(mu.transpose(-1,-2))
    aug1=th.cat((X,alpha*mu),2)
    aug2=th.cat((alpha*mu.transpose(1,2),th.ones(mu.shape[0],1,1,dtype=mu.dtype,device=f.device)),2)
    X=th.cat((aug1,aug2),1)
    if(reg_mode=='mle'):
        ret=X
    elif(reg_mode=='add_id'):
        ret=add_id(X,1e-6)
    elif(reg_mode=='adjust_eig'):
        ret=adjust_eig(0.75)(X)
    if(len(ret.shape)==3):
        return ret[:,None,:,:]
    return ret

def add_id(P,alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i]=P[i]+alpha*P[i].trace()*th.eye(P[i].shape[-1],dtype=P.dtype,device=P.device)
    return P

def dist_riemann(x,y):
    '''
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n) #wtf ca ne marche pas si je mets qu'une seule ?
    :return: tensor of values
    '''
    return LogEig.apply(CongrG(x,y,'neg')).view(x.shape[0],x.shape[1],-1).norm(p=2,dim=-1) #moyenne riemanniene si on minimise B par rapport a A_i

#cb
def dist_riemann_batches(x,y):
    '''
    Riemannian distance between SPD matrices x and SPD matrices y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: batch of SPD matrices (batch_size,1,n,n)
    :return:
    '''
    return th.linalg.matrix_norm(LogEig.apply(CongrG(x,y,'neg')))

#cb
def dist_riemann_matrix(x,y):
    '''
    Riemannian distance between SPD matrix x and SPD matrix y
    :param x: SPD matrices (n,n)
    :param y: SPD matrices (n,n)
    :return:
    '''
    return th.linalg.matrix_norm(LogEig.apply(CongrG(x,y[None,None,:,:],'neg')))

def CongrG(P,G,mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if(mode=='pos'):
        GG=SqmEig.apply(G) # avant : GG=SqmEig.apply(G[None,None,:,:]) # 
    elif(mode=='neg'):
        GG=SqminvEig.apply(G) # avant : GG=SqminvEig.apply(G[None,None,:,:]) #
    PP=GG.matmul(P).matmul(GG) #y^-1/2 x y^-1/2
    return PP

def LogG(x,X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x,X,'neg')),X,'pos')

def ExpG(x,X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x,X,'neg')),X,'pos')

def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size,channels,n=P.shape #batch size,channel depth,dimension
    Q=th.zeros(batch_size,channels,n,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i,j]=P[i,j].diag()
    return Q

def karcher_step(x,G,alpha):
    '''
    One step in the Karcher flow
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mean(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G

def BaryGeom(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k=1
    alpha=1
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:]
        for _ in range(k):
            G=karcher_step(x,G,alpha)
        return G

def karcher_step_weighted(x,G,alpha,weights):
    '''
    One step in the Karcher flow
    Weights is a weight vector of shape (batch_size,)
    Output is mean of shape (n,n)
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mul(weights[:,None,None,None]).sum(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G

def bary_geom_weighted(x,weights):
    '''
    Function which computes the weighted Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Weights is a weight vector of shape (batch_size,)
    Output is (1,1,n,n) Riemannian mean
    '''
    k=1
    alpha=1
    # with th.no_grad():
    G=x.mul(weights[:,None,None,None]).sum(dim=0)[0,:,:]
    for _ in range(k):
        G=karcher_step_weighted(x,G,alpha,weights)
    return G[None,None,:,:]

class Log_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.log(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 1/S

class Re_op():
    """ Log function and its derivative """
    _threshold=1e-4
    @classmethod
    def fn(cls,S,param=None):
        return nn.Threshold(cls._threshold,cls._threshold)(S)
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (S>cls._threshold).double()

class Sqm_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 0.5/th.sqrt(S)

class Sqminv_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return 1/th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return -0.5/th.sqrt(S)**3

class Power_op():
    """ Power function and its derivative """
    _power=1
    @classmethod
    def fn(cls,S,param=None):
        return S**cls._power
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (cls._power)*S**(cls._power-1)

class Inv_op():
    """ Inverse function and its derivative """
    @classmethod
    def fn(cls,S,param=None):
        return 1/S
    @classmethod
    def fn_deriv(cls,S,param=None):
        return log(S)

class Exp_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.exp(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return th.exp(S)
