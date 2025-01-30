#pipeline.py

import json
import argparse
from data_preprocessing import preprocess_data,load_data
from model import Autoencoder_SPDnet
from model_n_layers import Autoencoder_nlayers_SPDnet
from train import train
from test import test
from save_model import save_model
from spdnet.loss import RiemannianDistanceLoss
import torch
from pyriemann.classification import MDM
import warnings

def main():
    #load config
    parser = argparse.ArgumentParser(description='Train model')
    #parser.add_argument('dataset')
    parser.add_argument('-e', '--epochs', type=int , default = 5, help='Number of epochs for the training')
    parser.add_argument('-b','--batch_size', type=int , default = 32, help='Size of the batch for train/val/test')
    parser.add_argument('-r','--learning_rate', type=int , default = 0.01, help='Learning rate for the training')
    parser.add_argument('-d','--latent_dim', type=int , default = 2, help='Latent dimension of the autoencoder')
    parser.add_argument('-x', '--xp', type=int , default = 1, help='How many times the experience is repeated')
    parser.add_argument('-c', '--layers', type=int , default = 3, help='How many layers the model have')
    parser.add_argument('-n','--noise', default = 'none', help='Type of noise for the denoising. none if there is no noise.', choices=['none', 'gaussian', 'salt_pepper','masking'])
    parser.add_argument('-l','--loss', default = 'riemann', help='Loss. It can be riemannian or euclidean.', choices = ['euclidean','riemann'])
    args = parser.parse_args()
    #stored in : args.epochs, args.batch_size, args.learning_rate, args.latent_dim, args.noise , args.loss

    #load data
    X,labels = load_data()

    #preprocess data
    train_loader, val_loader, test_loader = preprocess_data(X,labels,batch_size=args.batch_size,noise=args.noise)
    #load model

    #train/val model
    ho, hi, ni, no = 1,1,X.data.shape[1],args.latent_dim
    auto_encoder = Autoencoder_nlayers_SPDnet(ho, hi, ni, no,args.layers)
    if args.loss == 'riemann':
        criterion = RiemannianDistanceLoss()
        mdm = MDM()
    else:
        criterion = torch.nn.MSELoss()
        mdm = MDM(metric="euclid")
    data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss = train(train_loader,val_loader,auto_encoder,args.epochs,criterion,noise=args.noise)

    #test model
    data_test,outputs_test,test_loss = test(test_loader,auto_encoder,criterion,noise=args.noise)

    # evaluate a un autre moment :
    # model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.eval()
    # ...

    #save_model
    save_model(auto_encoder,args.layers,args.loss,args.noise,args.epochs)

    #save datas

    pass

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()