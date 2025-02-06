import torch
import os
from visualization import show_first_image_from_loader,show_loss
import config as c


def find_second_folder_no_numero(epochs,latent_dim,latent_channel,loss,layers_type,data,synthetic_generation="",index=1,number_layers=0,batch_size=0,noise=""):
    second_folder = c.models_information_base_name
    second_folder += f"{c.models_information_n_epochs}{epochs}"
    second_folder += f"{c.models_information_latent_dim}{latent_dim}"
    second_folder += f"{c.models_information_latent_channel}{latent_channel}"
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

def find_name_folder(folder,epochs,latent_dim,latent_channel,loss,layers_type,data,synthetic_generation,index,number_layers,batch_size,noise):
    second_folder = find_second_folder_no_numero(epochs,latent_dim,latent_channel,loss,layers_type,data,synthetic_generation,index,number_layers,batch_size,noise)
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
    path = f"{folder}{c.models_information_model_name}{c.models_information_model_extension}"
    torch.save(model.state_dict(), path)

def save_datas_from_model(array,path,name):
    file=path+name+c.basic_extension
    torch.save(array,file)

def save_images_and_results(data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss,data_test,outputs_test,test_loss,test_trustworthiness,path,noised_train=None,noised_val=None,noised_test=None):    
    #images
    show_first_image_from_loader(data_train,path,name="original_train")
    show_first_image_from_loader(outputs_train,path,name="reconstruction_train")
    show_first_image_from_loader(data_val,path,name="original_val")
    show_first_image_from_loader(outputs_val,path,name="reconstruction_val")
    show_first_image_from_loader(data_test,path,name="original_test")
    show_first_image_from_loader(outputs_test,path,name="reconstruction_test")
    if noised_train is not None:
        show_first_image_from_loader(noised_train,path,name="noised_train")
        show_first_image_from_loader(noised_test,path,name="noised_test")
        show_first_image_from_loader(noised_val,path,name="noised_val")
    
    #save loss
    show_loss(list_train_loss,list_val_loss,path,name="loss_progression")
    #save datas format pt - pytorch
    save_datas_from_model(data_train,path,name="data_train")
    save_datas_from_model(outputs_train.detach(),path,name="outputs_train")
    save_datas_from_model(list_train_loss,path,name="list_train_loss")

    save_datas_from_model(data_val,path,name="data_val")
    save_datas_from_model(outputs_val.detach(),path,name="outputs_val")
    save_datas_from_model(list_val_loss,path,name="list_val_loss")

    save_datas_from_model(data_test,path,name="data_test")
    save_datas_from_model(outputs_test.detach(),path,name="outputs_test")

    save_datas_from_model(test_loss,path,name="test_loss")
    save_datas_from_model(test_trustworthiness,path,name="test_trustworthiness")

def save_synthetic_data(train_loader,val_loader,test_loader,name="block_diag"):
    file_path_train,file_path_val,file_path_test = find_name_dataset(name=name)
    torch.save(train_loader, file_path_train)
    torch.save(val_loader, file_path_val)
    torch.save(test_loader, file_path_test)

if __name__ == '__main__':
    path = find_name_folder("/models","one_layer",3,'riemann','none',5,16,2)
    print(path)