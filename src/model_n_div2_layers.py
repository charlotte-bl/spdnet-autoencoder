from spdnet import nn as spdnet
import torch.nn as nn
import torch

class Autoencoder_layers_byhalf_SPDnet(nn.Module):
    def __init__(self, ho, hi, ni, no):
        """
        Autoencoder pour les matrices SPD.
        Paramètres :
        - ho, hi : canaux de sortie et d'entrée pour BiMap
        - ni, no : dimensions des matrices d'entrée et de sortie
        - n_layers : nombre de couches dans l'encodeur et le décodeur
        """
        super(Autoencoder_layers_byhalf_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        
        #calcul tailles des couches intermediares
        self.layer_sizes = [ni]
        current_size = ni
        while current_size > self.no:
            current_size = max(self.no, int(current_size / 2))  # moitié mais jamais en dessous de no
            self.layer_sizes.append(current_size)

        # encoder
        encoder_layers = []
        for i in range(len(self.layer_sizes) - 1):
            encoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.layer_sizes[i],self.layer_sizes[i+1]))
            encoder_layers.append(spdnet.ReEig())
        encoder_layers.append(spdnet.LogEig())
        
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        decoder_layers.append(spdnet.ExpEig())
        decoder_layers.append(spdnet.ReEig())
        for i in range(len(self.layer_sizes) - 1, 0, -1):
            decoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.layer_sizes[i], self.layer_sizes[i-1]))
            decoder_layers.append(spdnet.ReEig())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction.double()

    def get_spd_parameters(self):
        return [self.other_param]
