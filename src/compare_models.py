import numpy as np
import config as c
from save import find_second_folder_results_influence_of_encoding_dim_name,find_path,find_comparison_path
from visualization import show_metrics_from_dicts
import os

def load_dict(path,name):
    # load a dictionnary
    data = np.load(path+name)
    loaded_keys = data['keys']
    loaded_means = data['means']
    loaded_stds = data['stds']
    reconstructed_dict = {k: (m, s) for k, m, s in zip(loaded_keys, loaded_means, loaded_stds)}
    return reconstructed_dict

def compare_channels_encoding(epochs, channels_out, losses, layers_type, data, noise, std=None, batch_size=None, generation=None, number_layers=None):
    #first folder to get data and second to where to stock comparisons in config.py
    folder = c.results_folder
    folder_comparison = find_comparison_path(epochs,data,noise,std,batch_size,generation)

    os.mkdir(folder_comparison)

    # dict to save loss in different files and not same graph bc not on the same scale
    list_dict_acc = []
    dict_dict_losses = {loss: [] for loss in losses}
    list_dict_trustworthiness = []

    # to save only one accuracy because always the same
    second_folder_acc_init = find_second_folder_results_influence_of_encoding_dim_name(epochs, channels_out[0], losses[0], layers_type[0], data, generation, number_layers, batch_size, noise, std)
    path_acc_init = find_path(folder, second_folder_acc_init, 1)
    acc_init = load_dict(path_acc_init, c.results_accuracy_init + c.extension_dict)
    list_dict_acc.append((acc_init, "Initial accuracy without AE"))

    # for each layers and each channels we want to put acc + trust + loss in the list/dict for the loss
    for layer in layers_type:
        for channel in channels_out:
            for loss in losses:
                second_folder = find_second_folder_results_influence_of_encoding_dim_name(epochs, channel, loss, layer, data, generation, number_layers, batch_size, noise, std)
                path = find_path(folder, second_folder, 1)

                acc_decode = load_dict(path, c.results_accuracy_decoding + c.extension_dict)
                list_dict_acc.append((acc_decode, f"Accuracy | channel : {channel} | layer : {layer} | loss : {loss} "))

                trustworthiness = load_dict(path, c.results_trustworthiness + c.extension_dict)
                list_dict_trustworthiness.append((trustworthiness, f"Trustworthiness | channel : {channel} | layer : {layer} | loss : {loss} "))

                loss_value = load_dict(path, c.results_losses + c.extension_dict)
                dict_dict_losses[loss].append((loss_value, f"Loss | channel : {channel} | layer : {layer} | loss : {loss}"))

    # create images
    show_metrics_from_dicts(list_dict_acc, path=folder_comparison + "/acc", xlabel="Encoding dimension", ylabel="Accuracy", title="Accuracy in function of encoding dimension")
    show_metrics_from_dicts(list_dict_trustworthiness, path=folder_comparison + "/trustworthiness", xlabel="Encoding dimension", ylabel="Trustworthiness", title="Trustworthiness in function of encoding dimension")
    for loss in losses:
        show_metrics_from_dicts(dict_dict_losses[loss], path=folder_comparison + "/loss_" + loss, xlabel="Encoding dimension", ylabel="Loss", title="Loss in function of encoding dimension")


if __name__ == '__main__':
    #fixed parameters of the model
    epochs = 200
    losses = [c.parsing_loss_riemann] #,c.parsing_loss_euclid]
    layers_type = ["regular"]
    data = "bci"
    number_layers=4
    batch_size=32
    noise="gaussian"
    std=0.01

    channels_out= [1, 2, 3, 4, 5,6,7,8]

    compare_channels_encoding(epochs=epochs,channels_out=channels_out,losses=losses,layers_type=layers_type,number_layers=number_layers,data=data,noise=noise,batch_size=batch_size,std=0.01)
