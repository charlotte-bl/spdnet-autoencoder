import torch
import torch.utils.data
from data_preprocessing import preprocess_data_cov
from save import save_synthetic_data
from scipy.stats import ortho_group
from scipy.linalg import block_diag
import numpy as np
import torch
from parsing import parsing_generation_data

# generation data based with one class

def generate_matrices_block_diag(number_matrices, size_matrices, lower_bound=3, upper_bound=6):
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

def generate_dataset_block_diag(X):
    labels = np.repeat(['unique_class'], X.shape[0])
    return X,labels

def generate_synthetic_data_block_diag(number_matrices,size_matrices,batch_size,noise,std):
    X = generate_matrices_block_diag(number_matrices=number_matrices, size_matrices=size_matrices)
    dataset,labels = generate_dataset_block_diag(X)
    return preprocess_data_cov(dataset,labels,batch_size,noise,std)

def generate_datasets_block_diag(number_dataset,number_matrices,size_matrices,batch_size,noise,std):
    for _ in range(number_dataset):
        train_loader, val_loader, test_loader = generate_synthetic_data_block_diag(number_matrices=number_matrices,size_matrices=size_matrices,batch_size=batch_size,noise=noise,std=std)
        save_synthetic_data(train_loader, val_loader, test_loader,name="block_diag")

# generation data based on lambda and mu

def generate_matrices_lambda_mu(number_matrices, size_matrices, lambda_mean=3, mu_mean=6):
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
    return X_sigma_plus.unsqueeze(dim=1).double(),X_sigma_minus.unsqueeze(dim=1).double() # on rajoute le channel

def generate_dataset_lambda_mu(X_sigma_plus,X_sigma_minus):
    labels_sigma_plus = np.repeat(['sigma_plus'], X_sigma_plus.shape[0])
    labels_sigma_minus = np.repeat(['sigma_minus'], X_sigma_minus.shape[0])
    return torch.cat((X_sigma_plus,X_sigma_minus)),np.concatenate((labels_sigma_plus,labels_sigma_minus))

def generate_synthetic_data_lambda_mu(number_matrices,size_matrices,batch_size,noise,std):
    X_sigma_plus,X_sigma_minus = generate_matrices_lambda_mu(number_matrices=number_matrices, size_matrices=size_matrices)
    dataset,labels = generate_dataset_lambda_mu(X_sigma_plus,X_sigma_minus)
    return preprocess_data_cov(dataset,labels,batch_size,noise,std)

def generate_datasets_lambda_mu(number_dataset,number_matrices,size_matrices,batch_size,noise,std):
    for _ in range(number_dataset):
        train_loader, val_loader, test_loader = generate_synthetic_data_lambda_mu(number_matrices=number_matrices,size_matrices=size_matrices,batch_size=batch_size,noise=noise,std=std)
        save_synthetic_data(train_loader, val_loader, test_loader,name="lambda_mu")

# generation of datasets

def main():
    args=parsing_generation_data()
    if args.synthetic_generation == "lambda_mu":
        generate_datasets_lambda_mu(number_dataset=args.number_dataset,number_matrices=args.number_matrices,size_matrices=args.size_block_matrices,batch_size=args.batch_size,noise=args.noise,std=args.std)
    else:
        generate_datasets_block_diag(number_dataset=args.number_dataset,number_matrices=args.number_matrices,size_matrices=args.size_block_matrices,batch_size=args.batch_size,noise=args.noise,std=args.std)

if __name__ == '__main__':
    main()