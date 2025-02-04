import torch
import torch.utils.data
from data_preprocessing import preprocess_data_cov
from save import save_synthetic_data
from scipy.stats import ortho_group
from scipy.linalg import block_diag
import numpy as np

def generate_matrices(number_matrices, size_matrices, lower_bound=3, upper_bound=6):
    X = torch.empty(0, size_matrices*2, size_matrices*2)
    U = torch.tensor(ortho_group.rvs(size_matrices)).float()
    b_inf,b_sup = lower_bound, upper_bound
    grid = torch.arange(b_inf,b_sup,(b_sup-b_inf)/size_matrices) #pas adapté pour être sur d'avoir une grid de size_matrices valeurs
    for _ in range(number_matrices):
        epsilon = torch.randn(size_matrices)*0.5
        X_i = U.T@ torch.diag(grid+epsilon) @U #création matrice SDP en fonction de val propres grid+eps et matrice ortho U
        X_i = (X_i.T + X_i) * 0.5 #symetrie
        X_i_diag = block_diag(X_i.numpy(),X_i.numpy()) #matrice carré avec deux carré qui sont X_i placés en diag et 0 sinon
        X_i_diag = X_i_diag / torch.linalg.matrix_norm(torch.from_numpy(X_i_diag)) #normalisation
        X = torch.cat((X, X_i_diag.unsqueeze(0)), dim=0) #rajout de la matrice au dataset
    return X.unsqueeze(dim=1).double()

def generate_dataset(X):
    labels = np.repeat(['unique_class'], X.shape[0])
    return X,labels

def generate_synthetic_data(number_matrices=300,size_matrices=25,batch_size=32,noise="none"):
    X = generate_matrices(number_matrices=number_matrices, size_matrices=size_matrices)
    dataset,labels = generate_dataset(X)
    #add noise if needed : not implemented
    return preprocess_data_cov(dataset,labels,batch_size,noise)

def generate_datasets(number_dataset=5):
    for _ in range(number_dataset):
        train_loader, val_loader, test_loader = generate_synthetic_data(size_matrices=8)
        save_synthetic_data(train_loader, val_loader, test_loader,name="block_diag")

if __name__ == '__main__':
    generate_datasets(number_dataset=1)

