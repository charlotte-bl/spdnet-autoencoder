#pipeline.py

import warnings
import os
import torch

from parsing import parsing_pipeline
from data_preprocessing import preprocess_data_BCI,load_data_BCI,load_preprocess_synthetic_data,get_size_matrix_from_loader,is_data_with_noise, dataloader_to_datasets

from models import Autoencoder_test_SPDnet, Autoencoder_nlayers_regular_SPDnet, Autoencoder_layers_byhalf_SPDnet, Autoencoder_one_layer_SPDnet

from sklearn.metrics import accuracy_score

from pyriemann.classification import MDM
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
    

    #### ENCAPSULATION A FAIRE DANS : metrics.py
    #convert loaders to numpy arrays to fit the MDM
    data_train_array,labels_train_array,decode_train_array,code_train_array,*optional_train_values = dataloader_to_datasets(train_loader,auto_encoder)
    data_val_array,labels_val_array,decode_val_array,code_val_array,*optional_val_values = dataloader_to_datasets(val_loader,auto_encoder)
    data_test_array,labels_test_array,decode_test_array,code_test_array,*optional_test_values = dataloader_to_datasets(test_loader,auto_encoder)
    if noised:
        noisy_data_train_array = optional_train_values[0]
        noisy_data_val_array = optional_val_values[0]
        noisy_data_test_array = optional_test_values[0]

    #train mdm
    mdm_init = MDM()
    mdm_init.fit(data_train_array,labels_train_array)
    y_pred_val = mdm_init.predict(data_val_array)
    y_pred_test = mdm_init.predict(data_test_array)

    acc_val = accuracy_score(labels_val_array,y_pred_val)


    print(y_pred_val,labels_val_array)
    mdm_decode = MDM()
    mdm_decode.fit(decode_train_array,labels_train_array)

    if auto_encoder.ho==1:
        mdm_code =  MDM()
        mdm_code.fit(code_train_array,labels_train_array)

    #############

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
        path,
        noisy_train=noisy_train if noised else None,  # if noised then put noise
        noisy_val=noisy_val if noised else None,     
        noisy_test=noisy_test if noised else None,  
        trustworthiness_encoding=trustworthiness_encoding if auto_encoder.ho == 1 else None
        )


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()