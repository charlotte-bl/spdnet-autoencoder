import torch.nn as nn
import torch
from torch.nn import Flatten
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier
import numpy as np


from SPDNet.optimizers import riemannian_adam as torch_riemannian_adam

from .layers import BiMap, LogEig, ReEig


class SPDNet_Module(nn.Module):
    def __init__(
        self, n_classes=2, bimap_dims=[64, 32], citerion=nn.CrossEntropyLoss()
    ):
        super(SPDNet_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        self.criterion = citerion
        layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims[1:]:
            layers.append(BiMap(1, 1, input_dim, output_dim))
            layers.append(ReEig())
            input_dim = output_dim
        layers.append(LogEig())
        layers.append(Flatten())
        lin_layer = nn.Linear(
            self.bimap_dims[-1] ** 2, self.n_classes, bias=False
        ).double()
        nn.init.xavier_uniform_(lin_layer.weight)
        layers.append(lin_layer)

        self.net = nn.Sequential(*layers)

    def fit(self, X, y, epochs=20, batch_size=64, shuffle=None):
        """
        shuffle is a mandatory paramters but not used
        """
        num_epochs = epochs
        X_train_tensor = torch.tensor(X, dtype=torch.float64)
        y_train_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch_riemannian_adam.RiemannianAdam(self.parameters(), lr=0.001)

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
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

    def predict(self, X):
        X_test_tensor = torch.tensor(X, dtype=torch.float64)
        test_dataset = TensorDataset(X_test_tensor, torch.zeros(X_test_tensor.shape))
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.eval()
        test_correct = 0
        y_pred = []
        with torch.no_grad():
            for inputs, t in test_dataloader:
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.append(np.array(predicted))

        return np.concatenate(y_pred)

    """
    def predict_proba(self, X):
        X_test_tensor = torch.tensor(X, dtype=torch.float64)
        test_dataset = TensorDataset(X_test_tensor, torch.zeros(X_test_tensor.shape))
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.eval()
        y_pred = []
        with torch.no_grad():
            for inputs, t in test_dataloader:
                outputs = self.forward(inputs)
                y_pred.append(np.array(outputs))

        return np.concatenate(y_pred)
    """

    def forward(self, x):
        x = x[:, None, :, :]
        y = self.net(x)
        return y


class SPDNet_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(SPDNet_Torch, self).__init__(**kwargs)
