import torch
import os

def find_name_model(number_layers,loss,noise,epochs):
    folder = "../models/"
    extension = ".pt"
    base_name = "model-"
    return find_name(folder,extension,base_name,number_layers,loss,noise,epochs)

def find_name_data(number_layers,loss,noise,epochs):
    folder = "../data/"
    extension = ".txt"
    base_name = "datas-"
    return find_name(folder,extension,base_name,number_layers,loss,noise,epochs)

def find_name(folder,extension,base_name,number_layers,loss,noise,epochs):
    name = base_name+"autoencoder_n-layers-"+str(number_layers)+"_loss-"+loss+"_noise-"+noise+"_n-epochs-"+str(epochs)
    index=1
    while True:
        new_name = f"{name}_{index:02d}{extension}"
        file_path = os.path.join(folder,new_name)
        if not os.path.exists(file_path):
            return file_path
        index = index+1

def save_model(model,number_layers,loss,noise,epochs):
    path = find_name_model(number_layers,loss,noise,epochs)
    torch.save(model.state_dict(), path)

def save_data():
    pass

if __name__ == '__main__':
    path = find_name_model(3,"riemann","none",45)
    print(path)