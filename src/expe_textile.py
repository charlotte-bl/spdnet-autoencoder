import numpy as np
import torch

from src.train import train
from models import Autoencoder_nlayers_regular_SPDnet
from src.spdnet.loss import RiemannianDistanceLoss
from src.data_preprocessing import LabeledDataset

from skimage import filters
import h5py
import pandas as pd

from tqdm import tqdm

from pyriemann.classification import MDM
from sklearn.model_selection import train_test_split


filename = "data/textile/train64.h5"
f = h5py.File(filename, "r")
my_data = pd.read_csv("data/textile/train64.csv", delimiter=",")

labels = my_data["indication_type"]
cool = f["images"]
ind = np.where((labels == "good") | (labels == "cut"))
coo = cool[ind[0], :, :, 0]
lab = labels[ind[0]]

Vals = []
Covs = []
imgs = []

for i in tqdm(range(len(ind[0]))):

    # We normalize the images to prevent any bias based on lighting or intensity
    # prior to filtering and covariance matrix computation.
    g = (coo[i, 1:, 1:] - np.mean(coo[i, 1:, 1:])) / np.std(coo[i, 1:, 1:])
    Intensity = g.flatten()
    Xdif = np.diff(g, axis=0)
    Ydif = np.diff(g, axis=1)
    Xdiff = np.diff(g, axis=0).flatten()
    Ydiff = np.diff(g, axis=1).flatten()
    gauss = filters.gaussian(g, sigma=2)
    gauss1 = filters.gaussian(g, sigma=3)
    gauss2 = filters.gaussian(g, sigma=4)
    gaus = filters.gaussian(g, sigma=2).flatten()
    gaus1 = filters.gaussian(g, sigma=3).flatten()
    gaus2 = filters.gaussian(g, sigma=4).flatten()
    hess = filters.laplace(g).flatten()
    frang = filters.farid(g).flatten()
    hess1 = filters.laplace(gauss).flatten()
    frang1 = filters.farid(gauss).flatten()
    hess2 = filters.laplace(gauss1).flatten()
    frang2 = filters.farid(gauss1).flatten()

    imgs.append(Intensity)

    Val = np.vstack(
        [Intensity, gaus, gaus1, gaus2, hess, frang, hess1, hess2, frang1, frang2]
    )

    Vals.append(Val)

    Covs.append(np.cov(Val))

all_cov = np.array(Covs)
all_labels = np.array(lab)
N, size_matrices, _ = all_cov.shape
all_cov_torch = torch.from_numpy(all_cov).unsqueeze(1)
dataset_textile = LabeledDataset(all_cov_torch, all_labels)

### Create dataloaders
batch_size = 32
x_train, x_val, x_test = torch.utils.data.random_split(
    dataset_textile, lengths=[0.5, 0.25, 0.25]
)

train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    x_val, batch_size=batch_size * 2, shuffle=False
)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)

### Create model
hi = 1  # Number of channels input
ho = 1  # Number of channels output
ni = size_matrices  # Dimension of matrices input
no = 4  # Dimension of matrices output
n_layers = 3

model = Autoencoder_nlayers_regular_SPDnet(ho, hi, ni, no, n_layers)

### Train model
criterion = RiemannianDistanceLoss()
n_epochs = 100
lr = 0.001

res = train(train_loader, val_loader, model, n_epochs, criterion, lr=lr)

### Train MDM before and after the AE

cov_test = x_test.dataset[x_test.indices][0].detach().numpy()[:, 0]
labels_test = x_test.dataset[x_test.indices][1]

# Create training and test splits
train_MDM, test_MDM, label_train_MDM, label_test_MDM = train_test_split(
    cov_test, labels_test, test_size=0.2, shuffle=True, random_state=42
)

# Fit the MDM before the AE
MDM_before = MDM()
MDM_before.fit(train_MDM, label_train_MDM)
MDM_before.score(test_MDM, label_test_MDM)

# Fit the MDM after the AE
model.eval()
with torch.no_grad():
    train_MDM_after_AE = model(torch.from_numpy(train_MDM).unsqueeze(1))
    test_MDM_after_AE = model(torch.from_numpy(test_MDM).unsqueeze(1))

MDM_after_AE = MDM()
MDM_after_AE.fit(train_MDM_after_AE.detach().numpy()[:, 0], label_train_MDM)
MDM_after_AE.score(test_MDM_after_AE.detach().numpy()[:, 0], label_test_MDM)
