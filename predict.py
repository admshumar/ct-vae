from models.vae_conv import ConvolutionalVAE
import argparse
import os
import json
import numpy as np


parser = argparse.ArgumentParser(description="A trained TensorFlow/Keras variational autoencoder for prediction.")
parser.add_argument("weight_directory", help="Directory containing the weights of the model.",
                    type=str)
parser.add_argument("--gpu", help="GPU to use for training. (Default: 0)",
                    type=int, default=0)
parser.add_argument("--number_of_predictions", help="Number of predictions to make. (Default: 1)",
                    type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

experiment_directory = os.path.abspath(os.path.join(os.getcwd(), 'data/experiments/vae_conv', args.weight_directory))
experiment_file = os.path.abspath(os.path.join(experiment_directory, 'experiment.json'))

with open(experiment_file, 'r') as file:
    experiment_dictionary = json.load(file)

vae = ConvolutionalVAE(is_mnist=experiment_dictionary['is_mnist'],
                       number_of_epochs=experiment_dictionary['number_of_epochs'],
                       enable_early_stopping=experiment_dictionary['enable_early_stopping'],
                       early_stopping_patience=experiment_dictionary['early_stopping_patience'],
                       # enable_logging=experiment_dictionary['enable_logging'],
                       enable_stochastic_gradient_descent=experiment_dictionary['enable_stochastic_gradient_descent'],
                       encoder_activation=experiment_dictionary['encoder_activation'],
                       decoder_activation=experiment_dictionary['decoder_activation'],
                       final_activation=experiment_dictionary['final_activation'],
                       # learning_rate_initial=experiment_dictionary['learning_rate_initial'],
                       # lr_reduction_patience=experiment_dictionary['lr_reduction_patience'],
                       beta=experiment_dictionary['beta'],
                       batch_size=experiment_dictionary['batch_size'],
                       latent_dimension=experiment_dictionary['latent_dimension'],
                       channel_size=experiment_dictionary['channel_size'],
                       depth=experiment_dictionary['depth'],
                       early_stopping_delta=experiment_dictionary['early_stopping_delta'])

# Load an autoencoder's weights
model = vae.load_model_weights(args.weight_directory)

# Specify data for the autoencoder
data = [vae.gaussian_test, vae.x_test]

# Get a prediction from the autoencoder
filename = os.path.abspath(os.path.join(vae.experiment_directory, 'ae_prediction.npy'))
reconstructed_data = vae.get_prediction(model, data=data, latent_only=True)
print(f"Saving file to {filename}.")
np.save(filename, reconstructed_data)

del vae
