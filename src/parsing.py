import argparse
import sys
import config as c

#pipeline.py
def parsing_pipeline():
    #load parsing argument
    parser = argparse.ArgumentParser(description='Train model')

    #model parameters
    parser.add_argument('-e', '--epochs', type=int , default = 5, help='Number of epochs for the training')
    parser.add_argument('-r','--learning_rate', type=int , default = 0.01, help='Learning rate for the training')
    parser.add_argument('-d','--encoding_dim', type=int , default = 2, help='Encoding dimension of the autoencoder')
    parser.add_argument('-o','--encoding_channel', type=int , default = 1, help='Encoding channel of the autoencoder')
    parser.add_argument('-l','--loss', default = c.parsing_loss_riemann, help='Loss. It can be riemannian or euclidean.', choices = [c.parsing_loss_riemann,c.parsing_loss_euclid])
    parser.add_argument('-m', '--layers_type',default='one_layer', help = 'How layers are implemented. Regular means layers are regular between input channels and output channels. By_halves means layers are reduced by half until no. If a layer is in dimension<10x10, then it is directly going to no.', choices = ['regular','by_halves','one_layer','hourglass_channel'])
    
    #visualization parameter
    parser.add_argument('-s', '--show', default=False,action='store_true')

    #datas and related arguments
    parser.add_argument('-j', '--data', default='synthetic', help ="Datas to train and test the autoencoder with.", choices = ['bci','synthetic'])
    parser.add_argument('-c', '--layers', type=int , default = 1, help='How many layers the model have. Only for the regular layers_type.')
    parser.add_argument('-b','--batch_size', type=int , default = 32, help='Size of the batch for train/val/test')
    parser.add_argument('-t', '--synthetic_generation', default='block_diag', help ="Which generation method to use for the model", choices = [c.parsing_synthetic_data_geodesics, c.parsing_synthetic_data_block_diag, c.parsing_synthetic_data_lambda_mu])
    parser.add_argument('-n','--noise', default = 'none', help='Type of noise for the denoising. none if there is no noise.', choices=['none', 'gaussian', 'salt_pepper','masking'])
    parser.add_argument('--std', default='0.01', type=float, help ="Standard deviation of the noise")
    parser.add_argument('-i', '--index', default='1', type=int, help ="Index of the synthetic data")
    
    args = parser.parse_args()

    #errors for dependencies of arguments
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
    if args.data=="synthetic" and "--batch_size" in sys.argv:
        parser.error("--batch_size has to be unspecified for the program to load the synthetic dataset")
    if args.data=="synthetic" and "--noise" in sys.argv:
        parser.error("--noise has to be unspecified for the program to load the synthetic dataset")
    if args.data=="synthetic" and "--std" in sys.argv:
        parser.error("--std has to be unspecified for the program to load the synthetic dataset")
    if args.noise=="none" and "--std" in sys.argv:
        parser.error("--std needs a noise")
    
    return args

#generate_synthetic_data.py
def parsing_generation_data():
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('-t', '--synthetic_generation', default='block_diag', help ="Which generation method to use for the model", choices = [c.parsing_synthetic_data_geodesics, c.parsing_synthetic_data_block_diag, c.parsing_synthetic_data_lambda_mu])
    parser.add_argument('-d', '--number_dataset', type=int , default = 1, help='Number of dataset you want to generate')
    parser.add_argument('-b','--batch_size', type=int , default = 32, help='Size of the batch for train/val/test')
    parser.add_argument('-s','--size_block_matrices', type=int , default = 8, help='Size of one class of matrices you want to have. Beware that the effective size of the matrix will be twice this value, since we have two classes.')
    parser.add_argument('-n','--noise', default = 'none', help='Type of noise for the denoising. none if there is no noise.', choices=['none', 'gaussian', 'salt_pepper','masking'])
    parser.add_argument('-q','--number_matrices', type=int , default = 300, help='How many matrices are in each of your dataset.')
    parser.add_argument('-e','--std', type=float , default = 0.01, help='Standard deviation of the noise if there is one.')

    args = parser.parse_args()

    if args.noise=="none" and "--std" in sys.argv:
        parser.error("--std has to be unspecified if there is no noise")

    return args
