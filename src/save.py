import torch
import os
from visualization import show_first_image_from_loader,show_loss
import config as c


def find_second_folder_no_numero(epochs,encoding_dim,encoding_channel,loss,layers_type,data,synthetic_generation="",index=1,number_layers=0,batch_size=0,noise="",std=1):
    second_folder = c.models_information_base_name
    second_folder += f"{c.models_information_n_epochs}{epochs}"
    second_folder += f"{c.models_information_encoding_dim}{encoding_dim}"
    second_folder += f"{c.models_information_encoding_channel}{encoding_channel}"
    second_folder += f"{c.models_information_loss}{loss}"
    second_folder += f"{c.models_information_layers_type}{layers_type}"
    second_folder += f"{c.models_information_data}{data}"
    if data=="synthetic":
        second_folder += f"{c.models_information_synthetic_generation}{synthetic_generation}"
        second_folder += f"{c.models_information_index}{index}"
    if layers_type=="regular":
        second_folder += f"{c.models_information_n_layers}{number_layers}"
    if data=="bci":
        second_folder += f"{c.models_information_batch_size}{batch_size}"
        second_folder += f"{c.models_information_noise}{noise}"
        if noise!="none":
            second_folder += f"{c.models_information_noise_std}{std}"
    return second_folder

def find_numero(folder,second_folder_no_index):
    index=1
    while True:
        new_second_folder = f"{second_folder_no_index}_{index:02d}/"
        folder_path = os.path.join(folder,new_second_folder)
        if not os.path.exists(folder_path):
            return index
        index = index+1

def find_path(folder,second_folder,numero):
    name = f"{folder}/{second_folder}_{numero:02d}/"
    return name

def find_result_path():
    folder = c.results_folder
    name = c.results_base_name
    index = find_numero(folder,name)
    return find_path(folder,name,index)

def find_name_folder(folder,epochs,encoding_dim,encoding_channel,loss,layers_type,data,synthetic_generation,index,number_layers,batch_size,noise,std):
    second_folder = find_second_folder_no_numero(epochs,encoding_dim,encoding_channel,loss,layers_type,data,synthetic_generation,index,number_layers,batch_size,noise,std)
    numero = find_numero(folder,second_folder)
    return find_path(folder,second_folder,numero)

def find_name_dataset(name="block_diag"):
    index=1
    folder = c.synthetic_data_folder
    name = c.synthetic_data_base_name+name+"_"
    name_train = name+"train"
    name_val = name+"val"
    name_test = name+"test"
    extension = c.synthetic_data_extension
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

def save_model(model,folder):
    path = folder+c.models_information_model_name+c.models_information_model_extension
    torch.save(model.state_dict(), path)


def save_datas_from_model(array,path,name):
    file=path+name+c.basic_extension
    torch.save(array,file)

def save_all(auto_encoder,data_train, outputs_train, list_train_loss, data_val, outputs_val, list_val_loss,
                            data_test, outputs_test, test_loss, trustworthiness, path,
                            noisy_train=None, noisy_val=None, noisy_test=None, trustworthiness_encoding=None):
    #show folder where to put the datas
    print(path)

    #save_model
    save_model(auto_encoder,path)
    #save images of covariances
    show_first_image_from_loader(data_train,path,name="last_batch_original_train")
    show_first_image_from_loader(outputs_train,path,name="last_batch_reconstruction_train")
    show_first_image_from_loader(data_val,path,name="last_batch_original_val")
    show_first_image_from_loader(outputs_val,path,name="last_batch_reconstruction_val")
    show_first_image_from_loader(data_test,path,name="last_batch_original_test")
    show_first_image_from_loader(outputs_test,path,name="last_batch_reconstruction_test")
    #save images of noised covariances if we put a noise
    if noisy_train is not None:
        show_first_image_from_loader(noisy_train,path,name="last_batch_noised_train")
        show_first_image_from_loader(noisy_val,path,name="last_batch_noised_test")
        show_first_image_from_loader(noisy_test,path,name="last_batch_noised_val")
    
    #save/show loss
    show_loss(list_train_loss,list_val_loss,path,name="loss_progression")
    save_datas_from_model(list_train_loss,path,name="list_train_loss")
    save_datas_from_model(list_val_loss,path,name="list_val_loss")
    save_datas_from_model(test_loss,path,name="test_loss")
    
    #save datas format pt - pytorch
    save_datas_from_model(data_train,path,name="last_batch_data_train")
    save_datas_from_model(outputs_train.detach(),path,name="last_batch_outputs_train")
    save_datas_from_model(data_val,path,name="last_batch_data_val")
    save_datas_from_model(outputs_val.detach(),path,name="last_batch_outputs_val")
    save_datas_from_model(data_test,path,name="last_batch_data_test")
    save_datas_from_model(outputs_test.detach(),path,name="last_batch_outputs_test")

    #save trustworthiness
    save_datas_from_model(trustworthiness,path,name="trustworthiness_decoding")
    if trustworthiness_encoding is not None:
        save_datas_from_model(trustworthiness_encoding,path,name="trustworthiness_encoding")

def save_synthetic_data(train_loader,val_loader,test_loader,name="block_diag"):
    #find name of the dataset
    file_path_train,file_path_val,file_path_test = find_name_dataset(name=name)
    #save loaders
    torch.save(train_loader, file_path_train)
    torch.save(val_loader, file_path_val)
    torch.save(test_loader, file_path_test)