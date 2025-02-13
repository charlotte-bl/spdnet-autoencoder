#name for synthetic datas
synthetic_data_folder = "../data/"
synthetic_data_base_name = "synthetic-dataloader_"
synthetic_data_extension = ".pt"

#parsing choices
parsing_loss_riemann = "riemann"
parsing_loss_euclid = "euclid"

parsing_synthetic_data_block_diag = "block_diag"
parsing_synthetic_data_lambda_mu = "lambda_mu"
parsing_synthetic_data_geodesics = "geodesics"


#name for folders with performance of a model
models_information_folder = "../models/"
models_information_base_name = "autoencoder"
models_information_model = "_models-"

#name for result folders
results_folder = "../evaluation_model/"
results_base_name = "experience"

models_information_n_epochs  = "_n-epochs-"
models_information_encoding_dim = "_encoding-dim-"
models_information_encoding_channel = "_channels-out-"
models_information_loss = "_loss-"
models_information_layers_type = "_layers-type-"
models_information_data = "_data-"
models_information_synthetic_generation = "_gen-"
models_information_index = "_index-"
models_information_n_layers = "_n-layers-"
models_information_batch_size = "_batch-size-"
models_information_noise = "_noise-"
models_information_noise_std = "_std-"

models_information_model_name = "model"
models_information_model_extension = ".pth"

#extension for other datas
basic_extension = ".pt"

#extension for other datas
extension_dict = ".npz"

#models data
model_trustworthiness_decoding = "trustworthiness_decoding"
model_test_loss = "test_loss"
model_accuracy_init = "accuracy_init"
model_accuracy_decoding = "accuracy_decoding"

#results information
results_accuracy_init = "dimension_accuracy_init"
results_accuracy_decoding = "dimension_accuracy_decoding"
results_losses = "dimension_losses"
results_trustworthiness = "dimension_trustworthiness"

#comparison models
comparison_folder = "../comparison_models/"
comparison_base_name="comparison"