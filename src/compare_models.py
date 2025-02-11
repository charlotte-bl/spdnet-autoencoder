import numpy as np

def load_dict(path,name):
    data = np.load(path+name)
    loaded_keys = data['keys']
    loaded_means = data['means']
    loaded_stds = data['stds']
    reconstructed_dict = {k: (m, s) for k, m, s in zip(loaded_keys, loaded_means, loaded_stds)}
    return reconstructed_dict

