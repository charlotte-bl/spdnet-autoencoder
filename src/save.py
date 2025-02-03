import torch
import os
from visualization import show_first_image_from_loader,show_loss

def find_name_folder(folder,number_layers,loss,noise,epochs,batch_size,layers_type):
    second_folder = "autoencoder_n-layers-"+str(number_layers)+"_loss-"+loss+"_noise-"+noise+"_n-epochs-"+str(epochs)+"_batch-size-"+str(batch_size)+"_layers-type-"+layers_type
    index=1
    while True:
        new_second_folder = f"{second_folder}_{index:02d}/"
        folder_path = os.path.join(folder,new_second_folder)
        if not os.path.exists(folder_path):
            return folder_path
        index = index+1

def save_model(model,folder):
    path = f"{folder}/model.pth"
    torch.save(model.state_dict(), path)

def save_data(data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss,data_test,outputs_test,test_loss,path,show=False):    
    show_first_image_from_loader(data_train,path,name="original_train")
    show_first_image_from_loader(outputs_train,path,name="reconstruction_train")
    show_first_image_from_loader(data_val,path,name="original_val")
    show_first_image_from_loader(outputs_val,path,name="reconstruction_val")
    show_first_image_from_loader(data_test,path,name="original_test")
    show_first_image_from_loader(outputs_test,path,name="reconstruction_test")
    show_loss(list_train_loss,list_val_loss,path,name="loss_progression")

if __name__ == '__main__':
    path = find_name_folder("/models",3,'riemann','none',5,16)
    print(path)