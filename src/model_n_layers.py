from spdnet import nn as spdnet
import torch.nn as nn
import torch

class Autoencoder_nlayers_SPDnet(nn.Module):
    def __init__(self, ho, hi, ni, no, n_layers):
        """
        Autoencoder pour les matrices SPD.
        Paramètres :
        - ho, hi : canaux de sortie et d'entrée pour BiMap
        - ni, no : dimensions des matrices d'entrée et de sortie
        - n_layers : nombre de couches dans l'encodeur et le décodeur
        """
        super(Autoencoder_nlayers_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        self.n_layers = n_layers
        
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        
        #calcul tailles des couches intermediares
        self.layer_sizes = [ni]
        for i in range(1, n_layers-1):
            size = (ni + no) * (n_layers-i) / n_layers
            self.layer_sizes.append(int(size))
        self.layer_sizes.append(no)

        # encoder
        encoder_layers = []
        for i in range(n_layers):
            encoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.ni if i == 0 else self.layer_sizes[i-1], self.layer_sizes[i]))
            self.ni_temp = self.layer_sizes[i]  # maj taille entrée de la prochaine couche
        encoder_layers.append(spdnet.ReEig())
        encoder_layers.append(spdnet.LogEig())
        
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        decoder_layers.append(spdnet.ExpEig())
        decoder_layers.append(spdnet.ReEig())
        for i in range(n_layers-1, 0, -1):
            decoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.layer_sizes[i], self.layer_sizes[i-1]))
            decoder_layers.append(spdnet.ReEig())
        decoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.layer_sizes[0], self.ni))
        decoder_layers.append(spdnet.ReEig())
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction.double()

    def get_spd_parameters(self):
        return [self.other_param]
