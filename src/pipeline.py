#pipeline.py

import warnings
import os
import torch

from parsing import parsing_pipeline
from data_preprocessing import preprocess_data_BCI,load_data_BCI,load_preprocess_synthetic_data,get_size_matrix_from_loader,is_data_with_noise, dataloader_to_datasets

from models import Autoencoder_nlayers_regular_SPDnet, Autoencoder_layers_byhalf_SPDnet, Autoencoder_one_layer_SPDnet

from metrics import accuracy
from spdnet.loss import RiemannianDistanceLoss
from train import train
from test import test
from save import find_name_folder
from save import save_all

def main():
    args = parsing_pipeline()
    
    #load data and preprocess data
    if args.data=="bci":
        X,labels = load_data_BCI()
        train_loader, val_loader, test_loader = preprocess_data_BCI(X,labels,batch_size=args.batch_size,noise=args.noise,std=args.std)
        ho, hi, ni, no = args.encoding_channel,1,X.data.shape[1],args.encoding_dim
    else:
        train_loader, val_loader, test_loader = load_preprocess_synthetic_data(args.index,args.synthetic_generation)
        ho, hi, ni, no = args.encoding_channel,1,get_size_matrix_from_loader(train_loader),args.encoding_dim
    noised = is_data_with_noise(train_loader)

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

    #train autoencoder
    data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss, *optional_values = train(train_loader,val_loader,auto_encoder,args.epochs,criterion)
    if noised:
        noisy_train,noisy_val = optional_values[0], optional_values[1]

    #test autoencoder
    data_test, outputs_test, test_loss, trustworthiness, *optional_values = test(test_loader,auto_encoder,criterion,show=args.show)
    if noised:
        noisy_test = optional_values[0]
    if auto_encoder.ho == 1:
        trustworthiness_encoding = optional_values[-1] #si il y a pas noise c'est 0 et si il y a c'est 1, c'est le dernier
    
    # accuracy of prediction
    acc_init,acc_decode,*optinal_value = accuracy(auto_encoder,train_loader,val_loader,test_loader)
    if auto_encoder.ho == 1:
        acc_code = optinal_value[0]

    #find folder name to save datas
    path = find_name_folder("../models",
                            args.epochs,
                            args.encoding_dim,
                            args.encoding_channel,
                            args.loss,
                            args.layers_type,
                            args.data,
                            args.synthetic_generation,
                            args.index,
                            args.layers,
                            args.batch_size,
                            args.noise,
                            args.std
                            )
    
    os.mkdir(path)

    #save datas
    save_all(
        auto_encoder,
        data_train,
        outputs_train,
        list_train_loss,
        data_val,
        outputs_val,
        list_val_loss,
        data_test,
        outputs_test,
        test_loss,
        trustworthiness,
        acc_init,
        acc_decode,
        path,
        noisy_train=noisy_train if noised else None,  # if noised then put noise
        noisy_val=noisy_val if noised else None,     
        noisy_test=noisy_test if noised else None,  
        trustworthiness_encoding=trustworthiness_encoding if auto_encoder.ho == 1 else None,
        acc_code=acc_code if auto_encoder.ho == 1 else None,
        )


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()