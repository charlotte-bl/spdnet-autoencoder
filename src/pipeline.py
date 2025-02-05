#pipeline.py

import warnings
import os
import torch

from parsing import parsing
from data_preprocessing import preprocess_data_BCI,load_data_BCI,load_preprocess_synthetic_data,get_size_matrix_from_loader

from models import Autoencoder_test_SPDnet, Autoencoder_nlayers_regular_SPDnet, Autoencoder_layers_byhalf_SPDnet, Autoencoder_one_layer_SPDnet

from pyriemann.classification import MDM
from spdnet.loss import RiemannianDistanceLoss
from train import train
from test import test
from save import find_name_folder
from save import save_model
from save import save_images_and_results


def main():
    args = parsing()
    
    #load data and preprocess data
    if args.data=="bci":
        X,labels = load_data_BCI()
        train_loader, val_loader, test_loader = preprocess_data_BCI(X,labels,batch_size=args.batch_size,noise=args.noise)
        ho, hi, ni, no = args.latent_channel,1,X.data.shape[1],args.latent_dim
    else:
        train_loader, val_loader, test_loader = load_preprocess_synthetic_data(args.index,args.synthetic_generation)
        ho, hi, ni, no = args.latent_channel,1,get_size_matrix_from_loader(train_loader),args.latent_dim

    #load model
    if args.layers_type == 'regular':
        auto_encoder = Autoencoder_nlayers_regular_SPDnet(ho, hi, ni, no,args.layers)
    elif args.layers_type == 'by_halves':
        auto_encoder = Autoencoder_layers_byhalf_SPDnet(ho, hi, ni, no)
    else:
        auto_encoder = Autoencoder_one_layer_SPDnet(ho, hi, ni, no)
    
    #loss
    if args.loss == 'riemann':
        criterion = RiemannianDistanceLoss()
    else:
        criterion = torch.nn.MSELoss()

    #train model
    data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss = train(train_loader,val_loader,auto_encoder,args.epochs,criterion)

    #test model
    if args.data=="bci":
        data_test,outputs_test,test_loss,test_trustworthiness = test(test_loader,auto_encoder,criterion,show=args.show,class_1_name=labels[0])
    else:
        data_test,outputs_test,test_loss,test_trustworthiness = test(test_loader,auto_encoder,criterion,show=args.show,class_1_name="")

    #find folder name to save datas
    path = find_name_folder("../models",args.layers_type,args.layers,args.loss,args.noise,args.epochs,args.batch_size,args.latent_dim)
    os.mkdir(path)
    
    #save_model
    save_model(auto_encoder,path)

    #save datas
    save_images_and_results(data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss,data_test,outputs_test,test_loss,test_trustworthiness,path,args.show)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()