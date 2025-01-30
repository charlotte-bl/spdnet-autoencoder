import torch
from pyriemann.estimation import Covariances
import moabb
from moabb.datasets import BNCI2014_001 # On s'interesse à ce dataset par exemple pour l'instant, il en existe pleins d'autres (cf site de MOABB)
from moabb.paradigms import FilterBankLeftRightImagery

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
        return noisy,clean, label

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

def raw_to_cov(raw_data):
    """
    Transform raw data to covarainces data
    :param raw_data: raw_data of shape (number_matrices, number_captors, number_data_by_captors) 
    :return: cova
    """
    return torch.from_numpy(Covariances(estimator='scm').fit_transform(raw_data)).unsqueeze(1)

def load_data():
    paradigm = FilterBankLeftRightImagery(filters =[[7,35]])
    dataset = BNCI2014_001()
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[8])
    return X,labels

def preprocess_data(X,labels,batch_size,noise):

    #covariances from data
    cov = raw_to_cov(X)

    #add noise if needed
    if noise=="salt_pepper":
        dataset = NoisyCleanLabeledDataset(cov,labels,add_salt_and_pepper_noise_to_covariances)
    elif noise=="masking":
        dataset = NoisyCleanLabeledDataset(cov,labels,add_masking_noise_to_covariances)
    elif noise=="gaussian":
        dataset = NoisyCleanLabeledDataset(cov,labels,add_gaussian_noise_to_covariances)
    else:
        dataset = LabeledDataset(cov,labels)
    x_train,x_val,x_test = torch.utils.data.random_split(dataset,lengths=[0.5,0.25,0.25])
    num_workers=0
    train_loader = torch.utils.data.DataLoader(x_train,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=True)

    val_loader = torch.utils.data.DataLoader(x_val,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            pin_memory=True)


    test_loader = torch.utils.data.DataLoader(x_test,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            pin_memory=True)
    return train_loader, val_loader, test_loader