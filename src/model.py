from spdnet import nn as spdnet
import torch.nn as nn
import torch

class Autoencoder_SPDnet(nn.Module):
    def __init__(self,ho,hi,ni,no,layers):
        """
        Autoencoder for SPD matrices.
        Parameters :
        - ho, hi : output and input channels for bimap
        - no, ni : output and input matrix dimensions
        """
        super(Autoencoder_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        self.encoder=nn.Sequential(
            spdnet.BiMap(self.ho,self.hi,self.ni,11),
            spdnet.BiMap(self.ho,self.hi,11,6),
            spdnet.BiMap(self.ho,self.hi,6,self.no),
            spdnet.ReEig(),
            spdnet.LogEig(),
        )
        self.decoder=nn.Sequential(
            spdnet.ExpEig(),
            spdnet.ReEig(),
            spdnet.BiMap(self.ho,self.hi,self.no,6),
            spdnet.ReEig(),
            spdnet.BiMap(self.ho,self.hi,6,11),
            spdnet.ReEig(),
            spdnet.BiMap(self.ho,self.hi,11,self.ni),
            spdnet.ReEig(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction.double()
    def get_spd_parameters(self):
        return [self.other_param]
        