import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional
from geoopt import Stiefel, ManifoldParameter

dtype=th.double
device=th.device('cpu')

class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """
    def __init__(self,ho,hi,ni,no):
        super().__init__()
        self._ni = ni
        self._no = no
        # We need to create arrays because the stiefel parameter can't optimiser for multiple channels at once
        if no > ni:
            self._W = [[ManifoldParameter(th.rand(no,ni ,dtype=dtype,device=device), manifold=Stiefel()) for _ in range(ho)] for _ in range(hi)]
        else:
            self._W = [[ManifoldParameter(th.rand(ni,no ,dtype=dtype,device=device), manifold=Stiefel()) for _ in range(ho)] for _ in range(hi)]
        self._ho=ho; self._hi=hi; self._ni=ni; self._no=no
        
        # Register the parameters
        with th.no_grad():
            for i in range(hi):
                for j in range(ho):
                    self._W[i][j].set_(self._W[i][j].manifold.random_naive(*self._W[i][j].shape, dtype=dtype, device=device))
                    self.register_parameter(f'W_{i}_{j}', self._W[i][j])
        
    def forward(self,X):
        return functional.bimap_channels(X,self._W, self._no > self._ni)

class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.LogEig.apply(P)

#cb
class ExpEig(nn.Module):
    """
    Author : cb
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of exp eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.ExpEig.apply(P)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.SqmEig.apply(P)

class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.ReEig.apply(P)

class BaryGeom(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    def forward(self,x):
        return functional.BaryGeom(x)

class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """
    def __init__(self,n):
        super(__class__,self).__init__()
        self.momentum=0.1
        self.register_buffer('running_mean', th.eye(n, dtype=dtype))
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight=functional.SPDParameter(th.eye(n,dtype=dtype))
    def forward(self,X):
        N,h,n,n=X.shape
        X_batched=X.permute(2,3,0,1).contiguous().view(n,n,N*h,1).permute(2,3,0,1).contiguous()
        if(self.training):
            mean=functional.BaryGeom(X_batched)
            with th.no_grad():
                self.running_mean.data=functional.geodesic(self.running_mean,mean,self.momentum)
            X_centered=functional.CongrG(X_batched,mean,'neg')
        else:
            X_centered=functional.CongrG(X_batched,self.running_mean,'neg')
        X_normalized=functional.CongrG(X_centered,self.weight,'pos')
        return X_normalized.permute(2,3,0,1).contiguous().view(n,n,N,h).permute(2,3,0,1).contiguous()

class CovPool(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        return functional.cov_pool(f,self._reg_mode)

class CovPoolBlock(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,L,n,T)
    Output X: L covariance matrices, shape (batch_size,L,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        ff=[functional.cov_pool(f[:,i,:,:],self._reg_mode)[:,None,:,:,:] for i in range(f.shape[1])]
        return th.cat(ff,1)

class CovPoolMean(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        return functional.cov_pool_mu(f,self._reg_mode)
    

class RiemannianDistanceLoss(nn.Module):
    """
    Input : 
    Output : Distance between
    Author : cb
    """
    def __init__(self):
        super(RiemannianDistanceLoss,self).__init__()
        self.distance = functional.dist_riemann_batches
    def forward(self,x,y):
        return self.distance(x,y).mean()
