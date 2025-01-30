
import torch.nn as nn
import torch
from torch.nn import Flatten, Conv2d,BatchNorm2d,MaxPool2d,Dropout,LeakyReLU
from skorch import NeuralNetClassifier

import sys
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts")
import TSMNet.spdnets.modules as modules

from .layers import BiMap, LogEig, ReEig, BatchNormSPD


class CNNSPDNetBN_Module(nn.Module):
    def __init__(self, n_channel_input, windows_size,n_classes=2, bimap_dims=[8,4], bn_momentum=0.1):
        super(CNNSPDNetBN_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        self.bn_momentum = bn_momentum


        layers = []
        # CNN part

        layers.append(Conv2d(1,
                16,
                kernel_size=(n_channel_input, 1),
                padding="valid",
                strides=(1, 1),
                activation=None,))
        layers.append(BatchNorm2d(1))
        layers.append(MaxPool2d(kernel_size=(2, 2), strides=(2, 2),
                padding="same"))
        layers.append(Dropout(0.5))
        
        layers.append(Conv2d(16,
                8,
                kernel_size=(1, 32),
                dilation_rate=(1, 2),
                padding="same",
            )
        )
        layers.append(BatchNorm2d(axis=1))
        layers.append(LeakyReLU(alpha=0.3))
        layers.append(MaxPool2d(kernel_size=(1, 2), strides=(1, 2), padding="same"))
        layers.append(Dropout(0.5))

        # layers.append(
        #     Conv2d(8,
        #         4,
        #         kernel_size=(5, 5),
        #         dilation_rate=(2, 2),
        #         padding="same",
        #     )
        # )
        # layers.append(BatchNorm2d(axis=1))
        # layers.append(LeakyReLU(alpha=0.3))
        # layers.append(MaxPool2d(kernel_size=(2, 2), padding="same"))
        # layers.append(Dropout(0.5))

        # Flatten to retrieve a 3D tensors
        layers.append(Flatten(start_dim=2))

        # Transform in SPD matrixes
        layers.append(torch.nn.Sequential(
            modules.CovariancePool(),
        ))

        # SPDBNnet part
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            layers.append(BiMap(1, 1, input_dim, output_dim))
            layers.append(BatchNormSPD(momentum=self.bn_momentum, n=output_dim))
            layers.append(ReEig())
            input_dim = output_dim
        layers.append(LogEig())
        layers.append(Flatten())
        lin_layer = nn.Linear(self.bimap_dims[-1] ** 2, self.n_classes, bias=False,dtype=torch.float64)#.double()
        nn.init.xavier_uniform_(lin_layer.weight)
        layers.append(lin_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, None, :, :]
        y = self.net(x)
        return y


class CNNSPDNetBN_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(CNNSPDNetBN_Torch, self).__init__(**kwargs)