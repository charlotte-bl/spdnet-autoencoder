import torch
import numpy as np
from visualization import show_metrics_from_dict
from save import find_path, find_second_folder_no_numero,find_result_path
import config as c
import os

def load_data_from_torch(path,name):
    data = torch.load(path+name,weights_only=True)
    return data

def compare_encoding_dim_influence():
    folder = c.models_information_folder
    folder_results = find_result_path()
    os.mkdir(folder_results)

    #fixed parameters of the model
    epochs = 50
    channels_out = 1
    loss_type = "riemann"
    layers_type = "one_layer"
    data = "synthetic"
    generation = "block_diag"
    number_layers=0
    batch_size=0
    noise=""
    std=1

    #parameters of the pipeline
    encoding_dims= [2, 4, 6, 8, 10, 12, 14, 16]
    nb_xp = 5
    nb_datasets = 5

    #where to keep the data

    dimension_losses = {dim:[] for dim in encoding_dims}
    dimension_trustworthiness = {dim:[] for dim in encoding_dims}
    for dim in encoding_dims:
        dataset_losses=[]
        dataset_trustworthiness=[]
        for dataset_index in range(1,nb_datasets+1):
            xp_losses=[]
            xp_trustworthiness=[]
            for repetition in range(1,nb_xp+1):
                model_second_folder = find_second_folder_no_numero(epochs,dim,channels_out,loss_type,layers_type,data,generation,dataset_index,number_layers,batch_size,noise,std)
                model_path = find_path(folder,model_second_folder,repetition)
                loss = load_data_from_torch(folder+model_path,"test_loss.pt")
                xp_losses.append(loss)
                trustworthiness = load_data_from_torch(folder+model_path,"test_trustworthiness.pt")
                xp_trustworthiness.append(trustworthiness)
            dataset_losses.append(xp_losses)
            dataset_trustworthiness.append(xp_trustworthiness)
        dimension_losses[dim] = (np.mean(dataset_losses), np.std(dataset_losses))
        dimension_trustworthiness[dim] = (np.mean(dataset_trustworthiness), np.std(dataset_trustworthiness))
    for dim in encoding_dims:
        print(f"Encoding dimension : {dim}")
        print(f"| Moyenne loss : {dimension_losses[dim][0]}, Écart-type loss : {dimension_losses[dim][1]}")
        print(f"| Moyenne trustworthiness : {dimension_trustworthiness[dim][0]}, Écart-type trustworthiness : {dimension_trustworthiness[dim][1]}")
    show_metrics_from_dict(dimension_losses,path=folder_results,name="dimension_losses",ylabel="Riemannian loss")
    show_metrics_from_dict(dimension_trustworthiness,path=folder_results,name="dimension_trustworthiness",ylabel="Trustworthiness")
    return dimension_losses,dimension_trustworthiness

if __name__ == '__main__':
    compare_encoding_dim_influence()
