import torch
import torch.utils.data
from data_preprocessing import preprocess_data_cov
from save import save_synthetic_data
from scipy.stats import ortho_group
from scipy.linalg import block_diag
import numpy as np

def generate_matrices(number_matrices=300, size_matrices=22, lambda_mean=3, mu_mean=6):
    X_sigma_plus = torch.empty(0, size_matrices*2, size_matrices*2)
    X_sigma_minus = torch.empty(0, size_matrices*2, size_matrices*2)
    U = torch.tensor(ortho_group.rvs(size_matrices)).float()
    for _ in range(number_matrices):
        epsilon = torch.abs(torch.randn(1))*0.3
        diag_lambda_flatten = torch.empty(0,1)
        diag_mu_flatten = torch.empty(0,1)
        for _ in range(size_matrices):
            value_lambda_diag=torch.normal(mean=lambda_mean,std=epsilon)
            diag_lambda_flatten =torch.cat((diag_lambda_flatten,value_lambda_diag.unsqueeze(0)),dim=0)
            diag_lambda = diag_lambda_flatten.squeeze(dim=1)

            value_mu_diag=torch.normal(mean=mu_mean,std=epsilon)
            diag_mu_flatten =torch.cat((diag_mu_flatten,value_mu_diag.unsqueeze(0)),dim=0)
            diag_mu = diag_mu_flatten.squeeze(dim=1)

        X_i_lambda = U.T@ torch.diag(diag_lambda) @U
        X_i_lambda = (X_i_lambda.T + X_i_lambda) * 0.5 #symetrie

        X_i_mu = U.T@ torch.diag(diag_mu) @U
        X_i_mu = (X_i_mu.T + X_i_mu) * 0.5 #symetrie

        sigma_plus_diag = torch.from_numpy(block_diag(X_i_lambda.numpy(),X_i_mu.numpy())) #matrice carré avec deux carré en diag, 0 sinon
        X_sigma_plus = torch.cat((X_sigma_plus, sigma_plus_diag.unsqueeze(0)), dim=0)

        sigma_minus_diag = torch.from_numpy(block_diag(X_i_mu.numpy(),X_i_lambda.numpy())) #matrice carré avec deux carré en diag, 0 sinon
        X_sigma_minus = torch.cat((X_sigma_minus, sigma_minus_diag.unsqueeze(0)), dim=0)
    
    return X_sigma_plus.unsqueeze(dim=1),X_sigma_minus.unsqueeze(dim=1) # on rajoute le channel

def generate_dataset(X_sigma_plus,X_sigma_minus):
    labels_sigma_plus = np.repeat(['sigma_plus'], X_sigma_plus.shape[0])
    labels_sigma_minus = np.repeat(['sigma_minus'], X_sigma_minus.shape[0])
    return torch.cat((X_sigma_plus,X_sigma_minus)),np.concatenate((labels_sigma_plus,labels_sigma_minus))

def generate_synthetic_data(number_matrices=300,size_matrices=25,batch_size=32,noise="none"):
    X_sigma_plus,X_sigma_minus = generate_matrices(number_matrices=number_matrices, size_matrices=size_matrices)
    dataset,labels = generate_dataset(X_sigma_plus,X_sigma_minus)
    return preprocess_data_cov(dataset,labels,batch_size,noise)

def generate_datasets(number_dataset=5):
    for _ in range(number_dataset):
        train_loader, val_loader, test_loader = generate_synthetic_data()
        save_synthetic_data(train_loader, val_loader, test_loader,name="lambda_mu")

if __name__ == '__main__':
    generate_datasets()