
import torch.nn as nn
import torch
from torch.nn import Flatten, Conv2d,BatchNorm2d,MaxPool2d,Dropout,LeakyReLU
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier
import numpy as np

import sys
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts")
from SPDNet.torch.optimizers import riemannian_adam as torch_riemannian_adam
import TSMNet.spdnets.modules as modules

from .layers import BiMap, LogEig, ReEig, BatchNormSPD


class SPDNetBN_Module(nn.Module):
    def __init__(self, n_classes=2, bimap_dims=[64, 32], bn_momentum=0.1, 
                 criterion=torch.nn.CrossEntropyLoss()):
        super(SPDNetBN_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        self.bn_momentum = bn_momentum
        self.criterion = criterion

        layers = []
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
        # print(x.shape)
        x = x[:, None, :, :]
        y = self.net(x)
        return y

    def fit(self,X,y,epochs=20,batch_size=64,shuffle=None):
        """
        shuffle is a mandatory paramters but not used
        """
        num_epochs = epochs
        X_train_tensor = torch.tensor(X, dtype=torch.float64)
        y_train_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer=torch_riemannian_adam.RiemannianAdam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = self.criterion(outputs.float(), labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

    def predict(self,X):
        X_test_tensor = torch.tensor(X, dtype=torch.float64)
        test_dataset = TensorDataset(X_test_tensor,torch.zeros(X_test_tensor.shape))
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.eval()
        test_correct = 0
        y_pred= []
        with torch.no_grad():
            for inputs,t in test_dataloader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.append(np.array(predicted))
                
        return np.concatenate(y_pred)


class SPDNetBN_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(SPDNetBN_Torch, self).__init__(**kwargs)


class CNNSPDNetBN_Module(nn.Module):
    def __init__(self, n_channel_input,n_classes=2, bimap_dims=[16,8,4], bn_momentum=0.1):
        super(CNNSPDNetBN_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        self.bn_momentum = bn_momentum
        self.criterion = torch.nn.CrossEntropyLoss()


        self.cnn_layers = []
        self.trans_layers = []
        # CNN part

        self.cnn_layers.append(Conv2d(1,
                16,
                kernel_size=(1, 16),
                padding="valid",
                stride=(1, 1),dtype=torch.float64))
        self.cnn_layers.append(BatchNorm2d(16,dtype=torch.float64))
        self.cnn_layers.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                padding=(1,1)))
        self.cnn_layers.append(Dropout(0.5))
        
        self.cnn_layers.append(Conv2d(16,
                8,
                kernel_size=(1, 32),
                dilation=(1, 2),
                padding="same",
                dtype=torch.float64,
            )
        )
        self.cnn_layers.append(BatchNorm2d(8,dtype=torch.float64))
        self.cnn_layers.append(LeakyReLU(0.3))
        self.cnn_layers.append(MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0,1)))
        self.cnn_layers.append(Dropout(0.5))

        self.cnn_layers.append(
            Conv2d(8,
                4,
                kernel_size=(5, 5),
                dilation=(2, 2),
                padding="same",
                dtype=torch.float64,
            )
        )
        self.cnn_layers.append(BatchNorm2d(4,dtype=torch.float64))
        self.cnn_layers.append(LeakyReLU(0.3))
        # self.cnn_layers.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1,1)))
        # self.cnn_layers.append(Dropout(0.5))
        # # self.cnn_layers = self.cnn_layers

        # # Flatten to retrieve a 3D tensors
        # self.trans_layers.append(Flatten(start_dim=2))

        # Transform in SPD matrixes
        self.trans_layers.append(modules.CovariancePool())
        # self.trans_layers = self.trans_layers

        # SPDBNnet part
        self.spd_layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            self.spd_layers.append(BiMap(1, 4, input_dim, output_dim,dtype=torch.float64))
            self.spd_layers.append(BatchNormSPD(momentum=self.bn_momentum, n=output_dim,dtype=torch.float64))
            self.spd_layers.append(ReEig())
            input_dim = output_dim
        self.spd_layers.append(LogEig())
        self.spd_layers.append(Flatten())
        print(self.bimap_dims[-1] ** 2, self.n_classes)
        lin_layer = nn.Linear(self.bimap_dims[-1] ** 2, self.n_classes, bias=False,dtype=torch.float64)#.double()
        nn.init.xavier_uniform_(lin_layer.weight)
        self.spd_layers.append(lin_layer)

        # self.spd_layers = self.spd_layers
    
        self.layer = self.cnn_layers.copy()
        self.layer += self.trans_layers
        self.layer += self.spd_layers

        self.net = nn.Sequential(*self.layer)

    def forward(self, x):
        # x = x[:, None, :, :]
        # y = self.net(x)
        t = x[:,None,...]
        # print(len(self.cnn_layers))
        # print(len(self.spd_layers))
        # print(len(self.layer))
        for i in range(len(self.cnn_layers)):
            # print("CNN layers {} with shape {}".format(self.cnn_layers[i],t.shape))
            t = self.cnn_layers[i](t).double()
        
        for i in range(len(self.trans_layers)):
            # print("trans layers {} with shape {}".format(self.cnn_layers[i],t.shape))
            t = self.trans_layers[i](t)
        # t = t[:,None,:,:]
        # print('add dimension',t.shape)
        # print(len(self.spd_layers))

        for i in range(len(self.spd_layers)):
            # print("SPD layers {} with shape {}".format(self.spd_layers[i],t.shape))
            t = self.spd_layers[i](t)

        
        y = t
        return y
    
    def fit(self,X,y,epochs=20,batch_size=64,shuffle=None):
        """
        shuffle is a mandatory paramters but not used
        """
        num_epochs = epochs
        X_train_tensor = torch.tensor(X, dtype=torch.float64)
        y_train_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer=torch_riemannian_adam.RiemannianAdam(self.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)
                loss = self.criterion(outputs.float(), labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

    def predict(self,X):
        X_test_tensor = torch.tensor(X, dtype=torch.float64)
        test_dataset = TensorDataset(X_test_tensor,torch.zeros(X_test_tensor.shape))
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.eval()
        test_correct = 0
        y_pred= []
        with torch.no_grad():
            for inputs,t in test_dataloader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.append(np.array(predicted))
                
        return np.concatenate(y_pred)


class CNNSPDNetBN_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(CNNSPDNetBN_Torch, self).__init__(**kwargs)