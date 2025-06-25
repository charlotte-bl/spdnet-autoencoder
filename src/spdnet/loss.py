import torch.nn as nn
from torch.autograd import Function as F
from . import functional

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
