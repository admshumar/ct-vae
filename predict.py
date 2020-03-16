from models.vae_conv import ConvolutionalVAE
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="A trained TensorFlow/Keras variational autoencoder for prediction.")
parser.add_argument("weight_directory", help="Directory containing the weights of the model.",
                    type=str)
parser.add_argument("-b", "--batch_size", help="Batch size. (Default: 5)",
                    type=int, default=5)
parser.add_argument("-c", "--channels", help="Number of channels for the first convolution map. (Default: 8)",
                    type=int, default=8)
parser.add_argument("-d", "--depth", help="Number downsampling/upsampling maps to apply in the encoder/decoder. (Default: 5)",
                    type=int, default=5)
parser.add_argument("-e", "--epochs", help="Number of epochs. (Default: 5)",
                    type=int, default=5)
parser.add_argument("-l", "--logging", help="Log training. (Default: True)",
                    type=bool, default=True)
parser.add_argument("--beta", help="Beta coefficient of the encoding loss. (Default: 1)",
                    type=float, default=1)
parser.add_argument("--dec_activation", help="Decoder activation function. (Default: 'relu')",
                    type=str, default='relu')
parser.add_argument("--early_stop", help="Train with early stopping. (Default: True)",
                    type=bool, default=True)
parser.add_argument("--early_stop_patience", help="Early stopping patience. (Default: 20)",
                    type=int, default=20)
parser.add_argument("--enc_activation", help="Encoder activation function. (Default: 'relu')",
                    type=str, default='relu')
parser.add_argument("--final_activation", help="Final activation function. (Default: 'sigmoid')",
                    type=str, default='sigmoid')
parser.add_argument("--gpu", help="GPU to use for training. (Default: 0)",
                    type=int, default=0)
parser.add_argument("--is_mnist", help="Use MNIST instead of a custom data set. (Default: False)",
                    type=bool, default=False)
parser.add_argument("--latent_dim", help="Dimension of the model's latent space. (Default: 2)",
                    type=int, default=2)
parser.add_argument("--learning_rate", help="Learning rate. (Default: 1e-4)",
                    type=float, default=1e-4)
parser.add_argument("--lr_plateau_patience", help="Factor for reducing learning rate upon plateau. (Default: 0.1)",
                    type=float, default=0.1)
parser.add_argument("--sgd", help="Train with stochastic gradient descent. (Default: True)",
                    type=bool, default=True)
parser.add_argument("--number_of_predictions", help="Number of predictions to make. (Default: 1)",
                    type=int, default=1)
args = parser.parse_args()

# weight_directory = os.path.abspath(os.path.join(os.getcwd(), 'data/experiments/vae_conv', args.weight_directory))
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

vae = ConvolutionalVAE(is_mnist=args.is_mnist,
                       number_of_epochs=args.epochs,
                       enable_early_stopping=args.early_stop,
                       early_stopping_patience=args.early_stop_patience,
                       enable_logging=args.logging,
                       enable_stochastic_gradient_descent=args.sgd,
                       encoder_activation=args.enc_activation,
                       decoder_activation=args.dec_activation,
                       final_activation=args.final_activation,
                       learning_rate_initial=args.learning_rate,
                       lr_reduction_patience=args.lr_plateau_patience,
                       beta=args.beta,
                       batch_size=args.batch_size,
                       latent_dimension=args.latent_dim,
                       channel_size=args.channels,
                       depth=args.depth,
                       early_stopping_delta=1e-2)

# Load an autoencoder's weights
model = vae.load_model_weights(args.weight_directory)

# Specify data for the autoencoder
data = [vae.gaussian_test, vae.x_test]

# Get a prediction from the autoencoder
reconstructed_data = vae.get_prediction(model, data=data, latent_only=True)

for i in range(args.number_of_predictions):
    image = vae.x_test[i]
    image = np.reshape(image, image.shape[0:2])
    plt.imsave(f'x_test_{i}.png', image)
    plt.close()

    image = reconstructed_data[i]
    image = np.reshape(image, image.shape[0:2])
    plt.imshow(image)
    plt.imsave(f'x_test_reconstructed_{i}.png', image)
    plt.close()

del vae
