import torch
import os
from visualization import show_first_image_from_loader,show_loss

def find_name_model(number_layers,loss,noise,epochs,batch_size):
    folder = "../models/"
    extension = ".pt"
    base_name = "model-"
    return find_name(folder,extension,base_name,number_layers,loss,noise,epochs,batch_size)

def find_name_data(number_layers,loss,noise,epochs,batch_size):
    folder = "../data/"
    extension = ".txt"
    base_name = "datas-"
    return find_name(folder,extension,base_name,number_layers,loss,noise,epochs,batch_size)

def find_name(folder,extension,base_name,number_layers,loss,noise,epochs,batch_size):
    name = base_name+"autoencoder_n-layers-"+str(number_layers)+"_loss-"+loss+"_noise-"+noise+"_n-epochs-"+str(epochs)+"_batch-size-"+str(batch_size)
    index=1
    while True:
        new_name = f"{name}_{index:02d}{extension}"
        file_path = os.path.join(folder,new_name)
        if not os.path.exists(file_path):
            return file_path
        index = index+1

def save_model(model,number_layers,loss,noise,epochs,batch_size):
    path = find_name_model(number_layers,loss,noise,epochs,batch_size)
    torch.save(model.state_dict(), path)

def save_data(data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss,data_test,outputs_test,test_loss,show=False):
    
    
    show_first_image_from_loader(data_train,show)
    show_first_image_from_loader(outputs_train,show)
    show_loss(list_train_loss,list_val_loss,show)

if __name__ == '__main__':
    path = find_name_model(3,"riemann","none",45)
    print(path)