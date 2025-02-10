from spdnet.functional import dist_riemann_matrix
import torch
import numpy as np
from data_preprocessing import dataloader_to_datasets
from sklearn.metrics import accuracy_score
from pyriemann.classification import MDM


def pairwise_riemannian_distances(batch):
    batch_size,_,n,_ = batch.shape
    batch = batch.squeeze(1)
    dist_matrix = torch.zeros(batch_size,batch_size,device=batch.device)
    for i in range(batch_size):
        for j in range(batch_size):
            dist_matrix[i,j] = dist_riemann_matrix(batch[i],batch[j]) #distances entre matrice i du batch vs matrice j du batch
    return dist_matrix

def pairwise_euclidean_distances(batch):
    batch_size,_,n,_ = batch.shape
    batch = batch.squeeze(1)
    dist_matrix = torch.zeros(batch_size,batch_size,device=batch.device)
    for i in range(batch_size):
        for j in range(batch_size):
            dist_matrix[i,j] = torch.norm(batch[i] - batch[j],p='fro') #distances entre matrice i du batch vs matrice j du batch
    return dist_matrix

def diag_inf(dist_matrix):
    dist_matrix.fill_diagonal_(float('inf'))
    return dist_matrix

def trustworthiness(original,reconstructed,k=2,pairwise_distance=pairwise_riemannian_distances):
    batch_size=original.shape[0]
    orig_dist = diag_inf(pairwise_distance(original)) #diag inf pour qu'une matrice n'ait pas elle meme comme plus proche voisin
    recon_dist = diag_inf(pairwise_distance(reconstructed))

    orig_ranks = torch.argsort(orig_dist, dim=-1)
    recon_ranks = torch.argsort(recon_dist, dim=-1) #r(i,j)
    #pour chaque matrice, on trie les k plus proches voisins
    
    differences = torch.zeros(batch_size, dtype=torch.float32)

    # premier signe somme
    for i in range(batch_size):
        # deuxieme signe somme : iterations des voisins
        for j in range(batch_size):
            if recon_ranks[i, j] < k: # pour que ce soit sur l'ensemble des k nearest neighbours dans l'espace output
                orig_rank = orig_ranks[i, j]
                recon_rank = recon_ranks[i, j]
                differences[i] += max(orig_rank - recon_rank - k, 0)


    numerator = 2 * differences.sum()
    denominator = batch_size * k * (2 * batch_size - 3 * k - 1)
    trustworthiness_score = 1 - (numerator / denominator)

    return trustworthiness_score.mean()

def accuracy(auto_encoder,train_loader,val_loader,test_loader):

    #convert loaders to numpy arrays to fit the MDM
    data_train_array,labels_train_array,decode_train_array,code_train_array,*optional_train_values = dataloader_to_datasets(train_loader,auto_encoder)
    data_val_array,labels_val_array,decode_val_array,code_val_array,*optional_val_values = dataloader_to_datasets(val_loader,auto_encoder)
    data_test_array,labels_test_array,decode_test_array,code_test_array,*optional_test_values = dataloader_to_datasets(test_loader,auto_encoder)

    # merge val and test to have more datas to train because we do not need validation for the MDM since we do not have any hyperparameters except the number of workers and the metric used (riemann/euclid)
    data_train_array = np.concatenate((data_train_array,data_val_array))
    labels_train_array = np.concatenate((labels_train_array,labels_val_array))
    decode_train_array = np.concatenate((decode_train_array,decode_val_array))
    code_train_array = np.concatenate((code_train_array,code_val_array))

    #train mdm
    # mdm with initial data
    mdm_init = MDM()
    mdm_init.fit(data_train_array,labels_train_array)
    y_pred_init = mdm_init.predict(data_test_array)
    acc_init = accuracy_score(labels_test_array,y_pred_init)

    # mdm after autoencoding
    mdm_decode = MDM()
    mdm_decode.fit(decode_train_array,labels_train_array)
    y_pred_decode = mdm_init.predict(decode_test_array)
    acc_decode = accuracy_score(labels_test_array,y_pred_decode)

    print("Accuracy score :")
    print(f"| Données initiales : {acc_init} ")
    print(f"| Données décodées : {acc_decode} ")
 
    result = [acc_init,acc_decode]

    if auto_encoder.ho==1:
        mdm_code =  MDM()
        mdm_code.fit(code_train_array,labels_train_array)
        y_pred_code = mdm_init.predict(code_test_array)
        acc_code = accuracy_score(labels_test_array,y_pred_code)
        print(f"| Données encodées : {acc_code} ")
        result.append(acc_code)
    return result

   

