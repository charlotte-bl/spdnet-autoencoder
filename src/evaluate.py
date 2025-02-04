import torch
from visualization import show_first_image
import matplotlib.pyplot as plt

def load_data_from_torch(path,name):
    data = torch.load(path+name,weights_only=True)
    return data

def test_image_data():
    path = "../models/autoencoder_models-one_layer_loss-riemann_noise-none_n-epochs-5_batch-size-32_latent-dim-8_01/"
    name = "data_train.pt"
    data =load_data_from_torch(path,name)
    plt.imshow(data.numpy()[0,0],cmap='gray')
    plt.colorbar()
    plt.show()

def test_list():
    path = "../models/autoencoder_models-one_layer_loss-riemann_noise-none_n-epochs-5_batch-size-32_latent-dim-8_01/"
    name = "list_train_loss.pt"
    data =load_data_from_torch(path,name)
    print(data)

def test_single_value():
    path = "../models/autoencoder_models-one_layer_loss-riemann_noise-none_n-epochs-5_batch-size-32_latent-dim-8_01/"
    name = "test_loss.pt"
    data =load_data_from_torch(path,name)
    name = "test_trustworthiness.pt"
    data2 =load_data_from_torch(path,name)
    print(data,data2)

if __name__ == '__main__':
    test_single_value()