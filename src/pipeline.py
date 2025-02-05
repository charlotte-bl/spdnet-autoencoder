#pipeline.py

import warnings
import os
import argparse
import torch
from data_preprocessing import preprocess_data_BCI,load_data_BCI,load_preprocess_synthetic_data,get_size_matrix_from_loader
from model_test import Autoencoder_test_SPDnet
from model_n_regular_layers import Autoencoder_nlayers_regular_SPDnet
from model_n_div2_layers import Autoencoder_layers_byhalf_SPDnet
from model_1_layer import Autoencoder_one_layer_SPDnet
from pyriemann.classification import MDM
from spdnet.loss import RiemannianDistanceLoss
from train import train
from test import test
from save import find_name_folder
from save import save_model
from save import save_images_and_results
import sys

def main():
    #load parsing argument
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-e', '--epochs', type=int , default = 5, help='Number of epochs for the training')
    parser.add_argument('-b','--batch_size', type=int , default = 32, help='Size of the batch for train/val/test')
    parser.add_argument('-r','--learning_rate', type=int , default = 0.01, help='Learning rate for the training')
    parser.add_argument('-d','--latent_dim', type=int , default = 2, help='Latent dimension of the autoencoder')
    parser.add_argument('-o','--latent_channel', type=int , default = 1, help='Latent channel of the autoencoder')
    parser.add_argument('-x', '--xp', type=int , default = 1, help='How many times the experience is repeated')
    parser.add_argument('-m', '--layers_type',default='one_layer', help = 'How layers are implemented. Regular means layers are regular between input channels and output channels. By_halves means layers are reduced by half until no. If a layer is in dimension<10x10, then it is directly going to no.', choices = ['regular','by_halves','one_layer'])
    parser.add_argument('-c', '--layers', type=int , default = 1, help='How many layers the model have')
    parser.add_argument('-n','--noise', default = 'none', help='Type of noise for the denoising. none if there is no noise.', choices=['none', 'gaussian', 'salt_pepper','masking'])
    parser.add_argument('-l','--loss', default = 'riemann', help='Loss. It can be riemannian or euclidean.', choices = ['euclidean','riemann'])
    parser.add_argument('-s', '--show', default=False,action='store_true')
    parser.add_argument('-j', '--data', default='synthetic', help ="Datas to train and test the autoencoder with.", choices = ['bci','synthetic'])
    parser.add_argument('-t', '--synthetic_generation', default='block_diag', help ="Which generation method to use for the model", choices = ['block_diag','lambda_mu'])
    parser.add_argument('-i', '--index', default='1', type=int, help ="Index of the synthetic data")
    args = parser.parse_args()
    #errors for dependencies
    if args.layers_type=="regular" and args.layers<=1:
        parser.error("--layers has to be greater than 1 for the layers to be with a regular step")
    if args.layers_type=="one_layer" and "--layers" in sys.argv:
        parser.error("--layers has to be unspecified for the model to have only one layer. layers type is currently 'one_layer'")
    if args.layers_type=="by_halves" and "--layers" in sys.argv:
        parser.error("--layers has to be unspecified for the model to cut dimension by halves at each layers")
    if args.data=="bci" and "--index" in sys.argv:
        parser.error("--index has to be unspecified for the program to load the BCI dataset")
    if args.data=="bci" and "--synthetic_generation" in sys.argv:
        parser.error("--synthetic_generation has to be unspecified for the program to load the BCI dataset")


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
    data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss = train(train_loader,val_loader,auto_encoder,args.epochs,criterion,noise=args.noise)

    #test model
    #data_test,outputs_test,test_loss = test(test_loader,auto_encoder,criterion,noise=args.noise,show=args.show,class_1_name=labels[0])
    data_test,outputs_test,test_loss,test_trustworthiness = test(test_loader,auto_encoder,criterion,noise=args.noise,show=args.show,class_1_name="")

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