import torch
import os
from visualization import show_first_image_from_loader,show_loss
import config as c

def find_name_folder(folder,layers_type,number_layers,loss,noise,epochs,batch_size,latent_dim):
    second_folder = c.models_information_base_name
    second_folder += f"{c.models_information_model}{layers_type}"
    if layers_type!="one_layer":
        second_folder += f"{c.models_information_n_layers}{number_layers}"
    second_folder += f"{c.models_information_loss}{loss}"
    second_folder += f"{c.models_information_noise}{noise}"
    second_folder += f"{c.models_information_n_epochs}{epochs}"
    second_folder += f"{c.models_information_batch_size}{batch_size}"
    second_folder += f"{c.models_information_latent_dim}{latent_dim}"    
    index=1
    while True:
        new_second_folder = f"{second_folder}_{index:02d}/"
        folder_path = os.path.join(folder,new_second_folder)
        if not os.path.exists(folder_path):
            return folder_path
        index = index+1

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

def save_images_and_results(data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss,data_test,outputs_test,test_loss,path,show=False):    
    show_first_image_from_loader(data_train,path,name="original_train")
    show_first_image_from_loader(outputs_train,path,name="reconstruction_train")
    show_first_image_from_loader(data_val,path,name="original_val")
    show_first_image_from_loader(outputs_val,path,name="reconstruction_val")
    show_first_image_from_loader(data_test,path,name="original_test")
    show_first_image_from_loader(outputs_test,path,name="reconstruction_test")
    show_loss(list_train_loss,list_val_loss,path,name="loss_progression")

def save_synthetic_data(train_loader,val_loader,test_loader,name="block_diag"):
    file_path_train,file_path_val,file_path_test = find_name_dataset(name=name)
    torch.save(train_loader, file_path_train)
    torch.save(val_loader, file_path_val)
    torch.save(test_loader, file_path_test)

if __name__ == '__main__':
    path = find_name_folder("/models","one_layer",3,'riemann','none',5,16,2)
    print(path)