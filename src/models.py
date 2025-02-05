from spdnet import nn as spdnet
import torch.nn as nn
import torch

class Autoencoder_test_SPDnet(nn.Module):
    def __init__(self,ho,hi,ni,no,layers):
        """
        Autoencoder for SPD matrices.
        Parameters :
        - ho, hi : output and input channels for bimap
        - no, ni : output and input matrix dimensions
        """
        super(Autoencoder_test_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        self.encoder=nn.Sequential(
            spdnet.BiMap(self.ho,self.hi,self.ni,11),
            spdnet.ReEig(),
            spdnet.BiMap(self.ho,self.hi,11,6),
            spdnet.ReEig(),
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

class Autoencoder_nlayers_regular_SPDnet(nn.Module):
    def __init__(self, ho, hi, ni, no, n_layers):
        """
        Autoencoder pour les matrices SPD.
        Paramètres :
        - ho, hi : canaux de sortie et d'entrée pour BiMap
        - ni, no : dimensions des matrices d'entrée et de sortie
        - n_layers : nombre de couches dans l'encodeur et le décodeur
        """
        super(Autoencoder_nlayers_regular_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        self.n_layers = n_layers
        
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        
        #calcul tailles des couches intermediares
        self.layer_sizes = [ni]
        step = (self.no - self.ni) / (self.n_layers - 1)
        for i in range(1, self.n_layers - 1):
            size = self.ni + int(i * step)
            self.layer_sizes.append(size)
        self.layer_sizes.append(self.no)

        # encoder
        encoder_layers = []
        for i in range(n_layers-1):
            encoder_layers.append(spdnet.BiMap(self.ho, self.hi, self.layer_sizes[i],self.layer_sizes[i+1]))
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
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def get_spd_parameters(self):
        return [self.other_param]

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

class Autoencoder_one_layer_SPDnet(nn.Module):
    def __init__(self,ho,hi,ni,no):
        """
        Autoencoder for SPD matrices.
        Parameters :
        - ho, hi : output and input channels for bimap
        - no, ni : output and input matrix dimensions
        """
        super(Autoencoder_one_layer_SPDnet, self).__init__()
        self.ho = ho
        self.hi = hi
        self.ni = ni
        self.no = no
        self.other_param = nn.Parameter(torch.randn(ho, hi, ni, no))
        self.encoder=nn.Sequential(
            spdnet.BiMap(self.ho,self.hi,self.ni,self.no),
            spdnet.ReEig(),
            spdnet.LogEig(),
        )
        self.decoder=nn.Sequential(
            spdnet.ExpEig(),
            spdnet.ReEig(),
            spdnet.BiMap(self.hi,self.ho,self.no,self.ni),
            spdnet.ReEig(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction.double()
    def get_spd_parameters(self):
        return [self.other_param]


