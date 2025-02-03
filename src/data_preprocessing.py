import torch
from pyriemann.estimation import Covariances
import moabb
from moabb.datasets import BNCI2014_001 # On s'interesse à ce dataset par exemple pour l'instant, il en existe pleins d'autres (cf site de MOABB)
from moabb.paradigms import FilterBankLeftRightImagery

# functions for noise
def add_gaussian_noise_to_covariances(cov):
    cov_noisy = torch.empty(0, 1, cov.shape[2], cov.shape[3])
    for i in range(cov.shape[0]):
        eigenvalues,eigenvectors = torch.linalg.eig(cov[i,0])
        epsilon = torch.randn(cov.shape[2])
        gamma = 5e1
        eigenvalues = eigenvalues.real + epsilon*gamma
        for j in range(eigenvalues.shape[0]):
            eigenvalues[j] = max(eigenvalues[j], 0.1)
        cov_noisy = torch.cat((cov_noisy, (eigenvectors @ torch.diag(eigenvalues.type(torch.cdouble)) @ torch.linalg.inv(eigenvectors)).unsqueeze(0).unsqueeze(0)) ,dim=0)
    return cov_noisy.real

def add_salt_and_pepper_noise_to_covariances(cov):
    return cov

def add_masking_noise_to_covariances(cov):
    return cov

# class to store the data to load the dataloader

class NoisyCleanDataset(torch.utils.data.Dataset):
    def __init__(self,data,add_noise=add_gaussian_noise_to_covariances):
        self.data = data
        self.noised_data = add_noise(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        noisy = self.noised_data[idx]
        clean = self.data[idx]
        return noisy,clean

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        data = self.data[idx]
        return data
    
class NoisyCleanLabeledDataset(torch.utils.data.Dataset):
    def __init__(self,data,labels,add_noise=add_gaussian_noise_to_covariances):
        self.data = data
        self.noised_data = add_noise(data)
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

# preprocessing for raw datas (such as BCI)

def raw_to_cov(raw_data):
    """
    Transform raw data to covarainces data
    :param raw_data: raw_data of shape (number_matrices, number_captors, number_data_by_captors) 
    :return: cova
    """
    return torch.from_numpy(Covariances(estimator='scm').fit_transform(raw_data)).unsqueeze(1)

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

def preprocess_data_cov_no_labels(X,batch_size,noise):
    if noise=="salt_pepper":
        dataset = NoisyCleanDataset(X,add_salt_and_pepper_noise_to_covariances)
    elif noise=="masking":
        dataset = NoisyCleanDataset(X,add_masking_noise_to_covariances)
    elif noise=="gaussian":
        dataset = NoisyCleanDataset(X,add_gaussian_noise_to_covariances)
    else:
        dataset = Dataset(X)
    x_train,x_val,x_test = torch.utils.data.random_split(dataset,lengths=[0.5,0.25,0.25])
    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size*2, shuffle=False)
    return train_loader, val_loader, test_loader

def preprocess_data_raw_no_labels(X_raw,batch_size,noise):
    cov = raw_to_cov(X_raw)
    return preprocess_data_cov_no_labels(cov,labels,batch_size,noise)

def preprocess_data_cov(X,labels,batch_size,noise):
    #add noise if needed
    if noise=="salt_pepper":
        dataset = NoisyCleanLabeledDataset(X,labels,add_salt_and_pepper_noise_to_covariances)
    elif noise=="masking":
        dataset = NoisyCleanLabeledDataset(X,labels,add_masking_noise_to_covariances)
    elif noise=="gaussian":
        dataset = NoisyCleanLabeledDataset(X,labels,add_gaussian_noise_to_covariances)
    else:
        dataset = LabeledDataset(X,labels)
    
    x_train,x_val,x_test = torch.utils.data.random_split(dataset,lengths=[0.5,0.25,0.25])

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size*2, shuffle=False)
    return train_loader, val_loader, test_loader

def preprocess_data_raw(X_raw,labels,batch_size,noise):
    cov = raw_to_cov(X_raw)
    return preprocess_data_cov(cov,labels,batch_size,noise)

def preprocess_data_BCI(X,labels,batch_size,noise):
    data_train_val,labels_train,data_test,labels_test = train_test_split_BCI(X,labels)

    #covariances from data
    cov_train_val = raw_to_cov(data_train_val)
    cov_test = raw_to_cov(data_test)

    #add noise if needed
    if noise=="salt_pepper":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_salt_and_pepper_noise_to_covariances)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_salt_and_pepper_noise_to_covariances)
    elif noise=="masking":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_masking_noise_to_covariances)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_masking_noise_to_covariances)
    elif noise=="gaussian":
        dataset_train_val = NoisyCleanLabeledDataset(cov_train_val,labels_train,add_gaussian_noise_to_covariances)
        x_test = NoisyCleanLabeledDataset(cov_test,labels_test,add_gaussian_noise_to_covariances)
    else:
        dataset_train_val = LabeledDataset(cov_train_val,labels_train)
        x_test = LabeledDataset(cov_test,labels_test)
    
    x_train,x_val = torch.utils.data.random_split(dataset_train_val,lengths=[0.8,0.2])

    train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size*2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    X,labels = load_data_BCI()
    train_test_split_BCI(X,labels)