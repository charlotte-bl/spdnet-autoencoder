import torch
import numpy as np
from visualization import show_metrics_from_dict
from save import find_path, find_second_folder_no_numero,find_result_path,save_dict_tuple
import config as c
import os

def load_data_from_torch(path,name):
    data = torch.load(path+name,weights_only=True)
    return data

def evaluate_encoding_dim_influence(epochs,encoding_dims,channels_out,loss_type,layers_type,data,noise,std=None,batch_size=None,generation=None,number_layers=None,nb_datasets=1,nb_xp=1):
    folder = c.models_information_folder #not to store but to get the other data

    folder_results = find_result_path(epochs,channels_out,loss_type,layers_type,data,generation,number_layers,batch_size,noise,std) #to store the results
    os.mkdir(folder_results)

    #where to keep the data
    dimension_losses = {dim:[] for dim in encoding_dims}
    dimension_trustworthiness = {dim:[] for dim in encoding_dims}
    dimension_accuracy_init = {dim:[] for dim in encoding_dims}
    dimension_accuracy_decode = {dim:[] for dim in encoding_dims}
    for dim in encoding_dims:
        dataset_losses=[]
        dataset_trustworthiness=[]
        dataset_acc_init = []
        dataset_acc_decode = []
        for dataset_index in range(1,nb_datasets+1):
            xp_losses=[]
            xp_trustworthiness=[]
            xp_acc_init = []
            xp_acc_decode = []
            for repetition in range(1,nb_xp+1):
                model_second_folder = find_second_folder_no_numero(epochs,dim,channels_out,loss_type,layers_type,data,generation,dataset_index,number_layers,batch_size,noise,std)
                model_path = find_path(folder,model_second_folder,repetition)
                loss = load_data_from_torch(folder+model_path,c.model_test_loss+c.basic_extension)
                xp_losses.append(loss)
                trustworthiness = load_data_from_torch(folder+model_path,c.model_trustworthiness_decoding+c.basic_extension)
                xp_trustworthiness.append(trustworthiness)
                acc_init = load_data_from_torch(folder+model_path,c.model_accuracy_init+c.basic_extension)
                xp_acc_init.append(acc_init)
                acc_decode = load_data_from_torch(folder+model_path,c.model_accuracy_decoding+c.basic_extension)
                xp_acc_decode.append(acc_decode)
            dataset_losses.append(xp_losses)
            dataset_trustworthiness.append(xp_trustworthiness)
            dataset_acc_init.append(xp_acc_init)
            dataset_acc_decode.append(xp_acc_decode)
        dimension_losses[dim] = (np.mean(dataset_losses), np.std(dataset_losses))
        dimension_trustworthiness[dim] = (np.mean(dataset_trustworthiness), np.std(dataset_trustworthiness))
        dimension_accuracy_init[dim] = (np.mean(dataset_acc_init), np.std(dataset_acc_init))
        dimension_accuracy_decode[dim] = (np.mean(dataset_acc_decode), np.std(dataset_acc_decode))
        
    for dim in encoding_dims:
        print(f"Encoding dimension : {dim}")
        print(f"| Moyenne loss : {dimension_losses[dim][0]}, Écart-type loss : {dimension_losses[dim][1]}")
        print(f"| Moyenne trustworthiness : {dimension_trustworthiness[dim][0]}, Écart-type trustworthiness : {dimension_trustworthiness[dim][1]}")
        print(f"| Moyenne accuracy decoding : {dimension_accuracy_decode[dim][0]}, Écart-type accuracy decoding : {dimension_accuracy_decode[dim][1]}")

    
    show_metrics_from_dict(dimension_losses,path=folder_results,name=c.results_losses,ylabel="Loss")
    show_metrics_from_dict(dimension_trustworthiness,path=folder_results,name=c.results_trustworthiness,ylabel="Trustworthiness")
    show_metrics_from_dict(dimension_accuracy_decode,path=folder_results,name=c.results_accuracy_decoding,ylabel="Decoding accuracy")

    save_dict_tuple(dimension_losses,path=folder_results,name=c.results_losses)
    save_dict_tuple(dimension_trustworthiness,path=folder_results,name=c.results_trustworthiness)
    save_dict_tuple(dimension_accuracy_init,path=folder_results,name=c.results_accuracy_init)
    save_dict_tuple(dimension_accuracy_decode,path=folder_results,name=c.results_accuracy_decoding)

    print(folder_results)

if __name__ == '__main__':
    #fixed parameters of the model
    epochs = 200
    channels_out = 2
    loss_type = "euclid"
    layers_type = "one_layer"
    data = "bci"
    batch_size=32
    noise="none"
    encoding_dims= [2, 4, 6, 8, 10, 12, 14, 16]
    nb_xp = 1
    nb_datasets = 1

    evaluate_encoding_dim_influence(epochs=epochs,encoding_dims=encoding_dims,channels_out=channels_out,loss_type=loss_type,layers_type=layers_type,data=data,noise=noise,batch_size=batch_size,nb_datasets=nb_datasets,nb_xp=nb_xp)
