from models.vae_conv import ConvolutionalVAE
import argparse
import os

parser = argparse.ArgumentParser(description="A TensorFlow/Keras variational autoencoder.")
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
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
vae = ConvolutionalVAE(is_mnist=False, #args.is_mnist,
                       number_of_epochs=2,#args.epochs,
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
                       latent_dimension=512, #args.latent_dim,
                       channel_size=args.channels,
                       depth=5,#args.depth,
                       early_stopping_delta=1e-2)
vae.train()
del vae
