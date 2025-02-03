import torch
import torch.utils.data
import os
from data_preprocessing import NoisyCleanDataset,Dataset,add_gaussian_noise_to_covariances,add_masking_noise_to_covariances,add_salt_and_pepper_noise_to_covariances
from scipy.stats import ortho_group
from scipy.linalg import block_diag

def generate_matrices(number_matrices, size_matrices, lower_bound=3, upper_bound=6):
    X = torch.empty(0, size_matrices*2, size_matrices*2)
    U = torch.tensor(ortho_group.rvs(size_matrices)).float()
    b_inf,b_sup = lower_bound, upper_bound
    grid = torch.arange(b_inf,b_sup,(b_sup-b_inf)/size_matrices) #pas adapté pour être sur d'avoir une grid de size_matrices valeurs
    for _ in range(number_matrices):
        epsilon = torch.randn(size_matrices)*0.2
        X_i = U.T@ torch.diag(grid+epsilon) @U #création matrice SDP en fonction de val propres grid+eps et matrice ortho U
        X_i = (X_i.T + X_i) * 0.5 #symetrie
        X_i_diag = block_diag(X_i.numpy(),X_i.numpy()) #matrice carré avec deux carré qui sont X_i placés en diag et 0 sinon
        X_i_diag = X_i_diag / torch.linalg.matrix_norm(torch.from_numpy(X_i_diag)) #normalisation
        X = torch.cat((X, X_i_diag.unsqueeze(0)), dim=0) #rajout de la matrice au dataset
    return X


def generate_synthetic_data(number_matrices=300,size_matrices=25,batch_size=32,noise="none"):
    x = generate_matrices(number_matrices=number_matrices, size_matrices=size_matrices)
    #add noise if needed
    if noise=="salt_pepper":
        dataset = NoisyCleanDataset(x,add_salt_and_pepper_noise_to_covariances)
    elif noise=="masking":
        dataset = NoisyCleanDataset(x,add_masking_noise_to_covariances)
    elif noise=="gaussian":
        dataset = NoisyCleanDataset(x,add_gaussian_noise_to_covariances)
    else:
        dataset = Dataset(x)
    x_train,x_val,x_test = torch.utils.data.random_split(dataset,lengths=[0.5,0.25,0.25])
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size*2, shuffle=False)
    return train_loader, val_loader, test_loader

def find_names_dataset():
    index=1
    folder = "../data"
    name = "synthetic-data_block_diag-loader_"
    name_train = name+"train"
    name_val = name+"val"
    name_test = name+"test"
    extension = ".pt"
    while True:
        new_name_train = f"{name_train}_{index:02d}{extension}"
        new_name_val = f"{name_val}_{index:02d}{extension}"
        new_name_test =f"{name_test}_{index:02d}{extension}"
        file_path_train = os.path.join(folder,new_name_train)
        file_path_val = os.path.join(folder,new_name_val)
        file_path_test = os.path.join(folder,new_name_test)
        if not os.path.exists(file_path_train):
            return file_path_train,file_path_val,file_path_test
        index = index+1

def generate_datasets(number_dataset=5):
    for i in range(number_dataset):
        train_loader, val_loader, test_loader = generate_synthetic_data()
        file_path_train,file_path_val,file_path_test = find_names_dataset()
        torch.save(train_loader, file_path_train)
        torch.save(val_loader, file_path_val)
        torch.save(test_loader, file_path_test)

if __name__ == '__main__':
    generate_datasets()

