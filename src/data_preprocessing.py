import torch
from pyriemann.estimation import Covariances
from moabb.datasets import BNCI2014_001 # On s'interesse à ce dataset par exemple pour l'instant, il en existe pleins d'autres (cf site de MOABB)
from moabb.paradigms import FilterBankLeftRightImagery
import config as c
import numpy as np

# functions for noise

def add_gaussian_noise_to_covariances(cov,std=1):
    cov_noisy = torch.empty(0, 1, cov.shape[2], cov.shape[3])
    for i in range(cov.shape[0]):
        scale = torch.max(torch.abs(cov[i][0]))
        epsilon = torch.randn(cov.shape[2], cov.shape[3]) * std * scale
        noise = epsilon @ epsilon.T
        
        cov_noisy =  torch.cat((cov_noisy, (cov[i][0] + noise).unsqueeze(0).unsqueeze(0)) ,dim=0)
    return cov_noisy.real

def add_gaussian_noise_eigen_decomposition(cov,std):
    cov_noisy = torch.empty(0, 1, cov.shape[2], cov.shape[3])
    for i in range(cov.shape[0]):
        eigenvalues,eigenvectors = torch.linalg.eig(cov[i,0])
        epsilon = torch.randn(cov.shape[2])
        eigenvalues = eigenvalues.real + epsilon*std
        for j in range(eigenvalues.shape[0]):
            eigenvalues[j] = max(eigenvalues[j], 0.1)
        cov_noisy = torch.cat((cov_noisy, (eigenvectors @ torch.diag(eigenvalues.type(torch.cdouble)) @ torch.linalg.inv(eigenvectors)).unsqueeze(0).unsqueeze(0)) ,dim=0)
    return cov_noisy.real

def add_salt_and_pepper_noise_to_covariances(cov,std):
    return cov

def add_masking_noise_to_covariances(cov,std):
    return cov

# manipulations on dataloader

def dataloader_to_datasets(dataloader,autoencoder):
    data_list,labels_list,noisy_data_list,decode_list,code_list = [],[],[],[],[]
    noised = is_data_with_noise(dataloader)

    for batch in dataloader:
        if noised:
            noisy_batch, data_batch,labels_batch = batch
            code_batch = autoencoder.encoder(noisy_batch)
            noisy_data_list.append(torch.squeeze(noisy_batch).numpy())
        else:
            data_batch,labels_batch = batch
            code_batch = autoencoder.encoder(data_batch)
        decode_batch = autoencoder.decoder(code_batch)
        data_list.append(torch.squeeze(data_batch).numpy())
        labels_list.append(labels_batch)
        decode_list.append(torch.squeeze(decode_batch.detach()).numpy())
        code_list.append(torch.squeeze(code_batch.detach()).numpy())

    data_array = np.concatenate(data_list,axis=0)
    labels_array = np.concatenate(labels_list,axis=0)
    decode_array = np.concatenate(decode_list,axis=0)
    code_array = np.concatenate(code_list,axis=0)
    results = [data_array,labels_array,decode_array,code_array]
    if noised:
        noisy_data_array = np.concatenate(noisy_data_list,axis=0)
        results.append(noisy_data_array)
    return results #[data_array,labels_array,decode_array,code_array,noisy_data_array]

# class to store the data to load the dataloader
    
class NoisyCleanLabeledDataset(torch.utils.data.Dataset):
    def __init__(self,data,labels,add_noise=add_gaussian_noise_to_covariances,std=1):
        self.data = data
        self.noised_data = add_noise(data,std)
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        noisy = self.noised_data[idx]
        clean = self.data[idx]
        label = self.labels[idx]
        return noisy,clean,label

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data = self.data[idx]
        label = self.label[idx]
        return data,label

#functional

def is_data_with_noise(loader):
    first_batch = next(iter(loader))
    return len(first_batch)==3

def get_size_matrix_from_loader(loader):
    if is_data_with_noise(loader):
        data, _ , _= next(iter(loader))
        input_size = data.shape
    else :
        data, _ = next(iter(loader))
        input_size = data.shape
    return input_size[2]

#load data

def load_preprocess_synthetic_data(index,name):
    path=c.synthetic_data_folder+c.synthetic_data_base_name+name
    path_train=f"{path}_train_{index:02d}{c.synthetic_data_extension}"
    path_val=f"{path}_val_{index:02d}{c.synthetic_data_extension}"
    path_test=f"{path}_test_{index:02d}{c.synthetic_data_extension}"
    train_loader = torch.load(path_train)
    val_loader = torch.load(path_val)
    test_loader = torch.load(path_test)
    return train_loader,val_loader,test_loader

def load_data_BCI():
    paradigm = FilterBankLeftRightImagery(filters =[[7,35]])
    dataset = BNCI2014_001()
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[8])
    return X,labels

def train_test_split_BCI(X,labels):
    #print(X.shape)
    #print(meta['session'].value_counts())
    X_train,labels_train = X[:144],labels[144:]
    X_test,labels_test = X[144:],labels[144:]
    return X_train,labels_train,X_test,labels_test

# preprocessing for raw datas (such as BCI)

def raw_to_cov(raw_data):
    """
    Transform raw data to covarainces data
    :param raw_data: raw_data of shape (number_matrices, number_captors, number_data_by_captors) 
    :return: cova
    """
    return torch.from_numpy(Covariances(estimator='scm').fit_transform(raw_data)).unsqueeze(1)

def preprocess_data_cov(X,labels,batch_size,noise,std):
    #add noise if needed
    if noise=="salt_pepper":
        dataset = NoisyCleanLabeledDataset(X,labels,add_salt_and_pepper_noise_to_covariances,std)
    elif noise=="masking":
        dataset = NoisyCleanLabeledDataset(X,labels,add_masking_noise_to_covariances,std)
    elif noise=="gaussian":
        dataset = NoisyCleanLabeledDataset(X,labels,add_gaussian_noise_to_covariances,std)
    elif noise=="gaussian_eigen":
        dataset = NoisyCleanLabeledDataset(X,labels,add_gaussian_noise_eigen_decomposition,std)
    else:
        dataset = LabeledDataset(X,labels)
    
    x_train,x_val,x_test = torch.utils.data.random_split(dataset,lengths=[0.5,0.25,0.25])

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size*2, shuffle=False)
    return train_loader, val_loader, test_loader

def preprocess_data_raw(X_raw,labels,batch_size,noise,std):
    cov = raw_to_cov(X_raw)
    return preprocess_data_cov(cov,labels,batch_size,noise,std)

def preprocess_data_BCI(X,labels,batch_size,noise,std):
    data_train_val,labels_train,data_test,labels_test = train_test_split_BCI(X,labels)

    #covariances from data
    cov_train_val = raw_to_cov(data_train_val)
    cov_test = raw_to_cov(data_test)

    #add noise if needed
    if noise=="salt_pepper":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_salt_and_pepper_noise_to_covariances,std)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_salt_and_pepper_noise_to_covariances,std)
    elif noise=="masking":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_masking_noise_to_covariances,std)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_masking_noise_to_covariances,std)
    elif noise=="gaussian":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_gaussian_noise_to_covariances,std)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_gaussian_noise_to_covariances,std)
    elif noise=="gaussian_eigen":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_gaussian_noise_eigen_decomposition,std)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_gaussian_noise_eigen_decomposition,std)
    else:
        dataset_train_val = LabeledDataset(cov_train_val,labels_train)
        x_test = LabeledDataset(cov_test,labels_test)
    
    x_train,x_val = torch.utils.data.random_split(dataset_train_val,lengths=[0.8,0.2])

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    #X,labels = load_data_BCI()
    #train_test_split_BCI(X,labels)
    train_loader,val_loader,test_loader = load_preprocess_synthetic_data(1,"block_diag")
    #print(len(train_loader))
    print(get_size_matrix_from_loader(train_loader))
    print(is_data_with_noise(train_loader))
