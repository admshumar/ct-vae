from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import numpy as np

import tensorflow
import tensorflow.keras.backend as k

from time import time

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, LeakyReLU

from utils import directories, logs, plots
from utils import classifiers, operations
from utils.plots import LatentSpaceTSNE
from utils.labels import OneHotEncoder, OneHotDecoder, Smoother, Flipper, GaussianSoftLabels
from utils.loaders import MNISTLoader, GenericLoader

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class VAE:
    """
    Base class for variational autoencoders, from which all autoencoder models inherit.
    """

    @classmethod
    def get_kwargs(cls):
        """
        Factory method for the default keyword arguments for the VAE class constructor.
        :return: A dictionary of boolean valued keyword arguments.
        """
        kwargs = dict()

        """
        Boolean valued keyword arguments
        """
        kwargs['deep'] = True
        kwargs['enable_activation'] = True
        kwargs['enable_augmentation'] = False
        kwargs['enable_batch_normalization'] = True
        kwargs['enable_dropout'] = True
        kwargs['enable_early_stopping'] = False
        kwargs['enable_logging'] = True
        kwargs['enable_label_smoothing'] = False
        kwargs['enable_rotations'] = False
        kwargs['enable_stochastic_gradient_descent'] = False
        kwargs['has_custom_layers'] = True
        kwargs['has_validation_set'] = False
        kwargs['is_mnist'] = True
        kwargs['is_restricted'] = False
        kwargs['is_standardized'] = False
        kwargs['show'] = False

        """
        Integer, float, and string valued keyword arguments.
        """
        kwargs['number_of_clusters'] = 3
        kwargs['restriction_labels'] = [1, 2, 3]
        kwargs['intermediate_dimension'] = 512
        kwargs['exponent_of_latent_space_dimension'] = 1
        kwargs['augmentation_size'] = 100
        kwargs['covariance_coefficient'] = 0.2
        kwargs['number_of_epochs'] = 5
        kwargs['batch_size'] = 128
        kwargs['learning_rate_initial'] = 1e-5
        kwargs['learning_rate_minimum'] = 1e-6
        kwargs['encoder_activation'] = 'relu'  # 'relu' 'tanh' 'elu' 'softmax' 'sigmoid'
        kwargs['decoder_activation'] = 'relu'
        kwargs['final_activation'] = 'sigmoid'
        kwargs['dropout_rate'] = 0.5
        kwargs['l2_constant'] = 1e-4
        kwargs['early_stopping_delta'] = 1
        kwargs['beta'] = 1
        kwargs['smoothing_alpha'] = 0.5
        kwargs['number_of_rotations'] = 2
        kwargs['angle_of_rotation'] = 30

        return kwargs

    @classmethod
    def shuffle(cls, data, labels):
        """
        Use NumPy's array indexing to shuffle two
        :param data:
        :param labels:
        :return:
        """
        assert len(data) == len(labels), f'Data and labels have incompatible shapes: {data.shape} and {labels.shape}'
        permutation = np.random.permutation(len(data))
        return data[permutation], labels[permutation]

    @classmethod
    def get_split_mnist_data(cls, val_size=0.5):
        """
        Grab the TensorFlow Keras incarnation of the MNIST data set, then split the training set into a training subset
        and a validation subset.
        :return: NumPy arrays of MNIST training, validation, and test sets.
        """
        (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size=val_size,
                                                        random_state=37,
                                                        stratify=y_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

    @classmethod
    def get_split_rotated_mnist_data(cls,
                                     restriction_labels,
                                     number_of_rotations,
                                     angle_of_rotation,
                                     angle_threshold=180,
                                     initial_split_size=0.4,
                                     val_size=0.5,
                                     is_confused=False,
                                     flipping_threshold=90):
        """
        Grab the TensorFlow Keras incarnation of the MNIST data set, then split the training set into a training subset
        and a validation subset.
        :return: NumPy arrays of MNIST training, validation, and test sets.
        """
        x_train = MNISTLoader('train', angle_threshold=angle_threshold).load(restriction_labels,
                                                                             number_of_rotations,
                                                                             angle_of_rotation)
        x_test = MNISTLoader('test', angle_threshold=angle_threshold).load(restriction_labels,
                                                                           number_of_rotations,
                                                                           angle_of_rotation)
        x_all = np.concatenate((x_train, x_test))
        y_train = MNISTLoader('train', angle_threshold=angle_threshold).load(restriction_labels,
                                                                             number_of_rotations,
                                                                             angle_of_rotation,
                                                                             label=True)
        y_test = MNISTLoader('test', angle_threshold=angle_threshold).load(restriction_labels,
                                                                           number_of_rotations,
                                                                           angle_of_rotation,
                                                                           label=True)
        y_all = np.concatenate((y_train, y_test))

        if is_confused:
            a_train = MNISTLoader('train', angle_threshold=angle_threshold).load(restriction_labels,
                                                                                 number_of_rotations,
                                                                                 angle_of_rotation,
                                                                                 angles=True)
            a_test = MNISTLoader('test', angle_threshold=angle_threshold).load(restriction_labels,
                                                                               number_of_rotations,
                                                                               angle_of_rotation,
                                                                               angles=True)
            a_all = np.concatenate((a_train, a_test))

            flipper = Flipper(y_all, a_all, threshold=flipping_threshold)

            y_all = flipper.change_labels()

            # x_all, y_all = VAE.shuffle(x_all, y_all)

            x_train, x_test, y_train, y_test, a_train, a_test = train_test_split(x_all, y_all, a_all,
                                                                                 test_size=initial_split_size,
                                                                                 random_state=37,
                                                                                 stratify=y_all)

            x_val, x_test, y_val, y_test, a_val, a_test = train_test_split(x_test, y_test, a_test,
                                                                           test_size=val_size,
                                                                           random_state=37,
                                                                           stratify=y_test)
            return x_train, y_train, a_train, x_val, y_val, a_val, x_test, y_test, a_test
        else:
            # x_all, y_all = VAE.shuffle(x_all, y_all)

            x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,
                                                                test_size=initial_split_size,
                                                                random_state=37,
                                                                stratify=y_all)

            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                            test_size=val_size,
                                                            random_state=37,
                                                            stratify=y_test)
            return x_train, y_train, x_val, y_val, x_test, y_test

    def __init__(self,
                 deep=True,
                 predict=False,
                 enable_activation=True,
                 enable_augmentation=False,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_early_stopping=False,
                 early_stopping_patience=20,
                 enable_lr_reduction=False,
                 lr_reduction_patience=10,
                 enable_logging=True,
                 enable_manual_clusters=False,
                 enable_rotations=False,
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 has_validation_set=True,
                 validation_size=0.5,
                 is_confused=False,
                 is_mnist=True,
                 is_restricted=False,
                 is_standardized=False,
                 show=False,
                 with_mixture_model=False,
                 with_logistic_regression=False,
                 with_svc=False,
                 number_of_clusters=3,
                 restriction_labels=list(range(10)),
                 intermediate_dimension=512,
                 latent_dimension=2,
                 exponent_of_latent_space_dimension=1,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 number_of_epochs=5,
                 batch_size=128,
                 learning_rate_initial=1e-5,
                 learning_rate_minimum=1e-6,
                 dropout_rate=0.5,
                 l2_constant=1e-4,
                 early_stopping_delta=0.01,
                 beta=1,
                 smoothing_alpha=0.0,
                 number_of_rotations=11,
                 angle_of_rotation=30,
                 angle_threshold=180,
                 encoder_activation='relu',
                 decoder_activation='relu',
                 encoder_activation_layer=LeakyReLU(),
                 decoder_activation_layer=LeakyReLU(),
                 final_activation='sigmoid',
                 model_name='vae',
                 from_vae=True):
        self.predict = predict
        self.model_name = model_name
        self.enable_logging = enable_logging
        self.enable_label_smoothing = (smoothing_alpha > 0)
        self.deep = deep
        self.is_confused = is_confused
        self.is_mnist = is_mnist
        self.is_restricted = is_restricted
        self.is_thresholded = (angle_threshold != 180)
        self.restriction_labels = restriction_labels
        self.enable_early_stopping = enable_early_stopping and has_validation_set
        self.enable_rotations = enable_rotations
        self.number_of_rotations = number_of_rotations
        self.angle_of_rotation = angle_of_rotation
        self.with_mixture_model = with_mixture_model
        self.with_logistic_regression = with_logistic_regression
        self.with_svc = with_svc
        self.alpha = smoothing_alpha
        self.validation_size = validation_size
        self.angle_threshold = angle_threshold

        self.is_standardized = is_standardized
        self.enable_stochastic_gradient_descent = enable_stochastic_gradient_descent
        self.has_custom_layers = has_custom_layers
        self.exponent_of_latent_space_dimension = exponent_of_latent_space_dimension
        self.enable_augmentation = enable_augmentation
        self.augmentation_size = augmentation_size
        self.covariance_coefficient = covariance_coefficient
        self.show = show
        self.restriction_labels = restriction_labels
        self.early_stopping_patience = early_stopping_patience
        self.enable_lr_reduction = enable_lr_reduction
        self.lr_reduction_patience = lr_reduction_patience

        self.has_validation_set = has_validation_set

        if from_vae:
            pass
            # x_train, y_train, x_val, y_val, x_test, y_test = VAE.get_split_mnist_data()
        if is_mnist:
            if has_validation_set:
                x_train, y_train, x_val, y_val, x_test, y_test = VAE.get_split_mnist_data()

                if is_restricted:
                    x_train, y_train = operations.restrict_data_by_label(x_train, y_train, restriction_labels)
                    x_val, y_val = operations.restrict_data_by_label(x_val, y_val, restriction_labels)
                    x_test, y_test = operations.restrict_data_by_label(x_test, y_test, restriction_labels)
                    if enable_rotations:
                        print("Rotations enabled!")
                        if is_confused:
                            x_train, y_train, a_train, \
                            x_val, y_val, a_val, \
                            x_test, y_test, a_test = VAE.get_split_rotated_mnist_data(restriction_labels,
                                                                                      number_of_rotations,
                                                                                      angle_of_rotation,
                                                                                      angle_threshold=angle_threshold,
                                                                                      is_confused=True)
                            self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
                            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
                            self.a_train, self.a_val, self.a_test = a_train, a_val, a_test
                        else:
                            x_train, y_train, \
                            x_val, y_val, \
                            x_test, y_test = VAE.get_split_rotated_mnist_data(restriction_labels,
                                                                              number_of_rotations,
                                                                              angle_of_rotation,
                                                                              angle_threshold=angle_threshold,
                                                                              is_confused=False)
                else:
                    if enable_rotations:
                        print("Rotations enabled!")
                        x_train, y_train, \
                        x_val, y_val, \
                        x_test, y_test = VAE.get_split_rotated_mnist_data(list(range(10)),
                                                                          number_of_rotations,
                                                                          angle_of_rotation,
                                                                          angle_threshold=angle_threshold,
                                                                          is_confused=is_confused)

                self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test \
                    = x_train, y_train, x_val, y_val, x_test, y_test

                self.y_train_binary = OneHotEncoder(y_train).encode()
                self.y_val_binary = OneHotEncoder(y_val).encode()
                self.y_test_binary = OneHotEncoder(y_test).encode()

                if smoothing_alpha > 0:
                    self.y_train_smooth = Smoother(y_train, alpha=smoothing_alpha).smooth()
                    self.y_val_smooth = Smoother(y_val, alpha=smoothing_alpha).smooth()
                    self.y_test_smooth = Smoother(y_test, alpha=smoothing_alpha).smooth()

            else:
                (x_train, y_train), (x_test, y_test) = mnist.load_data()

                if is_restricted:
                    x_train, y_train = operations.restrict_data_by_label(x_train, y_train, restriction_labels)
                    x_test, y_test = operations.restrict_data_by_label(x_test, y_test, restriction_labels)

                if enable_rotations:
                    print("Rotations enabled!")
                    x_train = MNISTLoader('train', angle_threshold=self.angle_threshold).load(restriction_labels,
                                                                                              number_of_rotations,
                                                                                              angle_of_rotation)
                    y_train = MNISTLoader('train', angle_threshold=self.angle_threshold).load(restriction_labels,
                                                                                              number_of_rotations,
                                                                                              angle_of_rotation,
                                                                                              label=True)
                    x_train, y_train = VAE.shuffle(x_train, y_train)

                    x_test = MNISTLoader('test', angle_threshold=self.angle_threshold).load(restriction_labels,
                                                                                            number_of_rotations,
                                                                                            angle_of_rotation)
                    y_test = MNISTLoader('test', angle_threshold=self.angle_threshold).load(restriction_labels,
                                                                                            number_of_rotations,
                                                                                            angle_of_rotation,
                                                                                            label=True)
                    x_test, y_test = VAE.shuffle(x_test, y_test)

                self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

                self.y_train_binary = OneHotEncoder(y_train).encode()
                self.y_test_binary = OneHotEncoder(y_test).encode()

                if smoothing_alpha > 0:
                    self.y_train_smooth = Smoother(y_train, alpha=smoothing_alpha).smooth()
                    self.y_test_smooth = Smoother(y_test, alpha=smoothing_alpha).smooth()

            if is_restricted:
                self.number_of_clusters = len(restriction_labels)
            else:
                self.number_of_clusters = len(np.unique(y_train))

            self.enable_manual_clusters = enable_manual_clusters
            if enable_manual_clusters:
                self.number_of_clusters = number_of_clusters

            # self.data_width, self.data_height = self.x_train.shape[1], self.x_train.shape[2]
            # self.data_dimension = self.data_width * self.data_height
            # self.intermediate_dimension = intermediate_dimension

            self.x_train = operations.normalize(self.x_train)
            self.x_test = operations.normalize(self.x_test)
            if has_validation_set:
                self.x_val = operations.normalize(self.x_val)

            self.gaussian_train = operations.get_gaussian_parameters(self.x_train, latent_dimension)
            self.gaussian_test = operations.get_gaussian_parameters(self.x_test, latent_dimension)
            if has_validation_set:
                self.gaussian_val = operations.get_gaussian_parameters(self.x_val, latent_dimension)
        else:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = GenericLoader(
                'chest_xray').load()
            self.x_train = operations.normalize(self.x_train)
            self.x_test = operations.normalize(self.x_test)
            if has_validation_set:
                self.x_val = operations.normalize(self.x_val)
            self.gaussian_train = operations.get_gaussian_parameters(self.x_train, latent_dimension)
            self.gaussian_test = operations.get_gaussian_parameters(self.x_test, latent_dimension)
            if has_validation_set:
                self.gaussian_val = operations.get_gaussian_parameters(self.x_val, latent_dimension)

        self.x_train_length = len(self.x_train)
        self.x_test_length = len(self.x_test)
        self.data_width, self.data_height = self.x_train.shape[1], self.x_train.shape[2]
        self.data_dimension = self.data_width * self.data_height
        self.intermediate_dimension = intermediate_dimension

        """
        Hyperparameters for the neural network.
        """
        self.number_of_epochs = number_of_epochs

        if self.enable_stochastic_gradient_descent:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.x_train)

        self.learning_rate = learning_rate_initial
        self.learning_rate_minimum = learning_rate_minimum
        self.enable_batch_normalization = enable_batch_normalization
        self.enable_dropout = enable_dropout
        self.enable_activation = enable_activation
        self.encoder_activation = encoder_activation  # 'relu', 'tanh', 'elu', 'softmax', 'sigmoid'
        self.decoder_activation = decoder_activation
        self.encoder_activation_layer = encoder_activation_layer
        self.decoder_activation_layer = decoder_activation_layer
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.l2_constant = l2_constant
        self.early_stopping_delta = early_stopping_delta

        self.latent_dimension = latent_dimension
        self.gaussian_dimension = 2 * self.latent_dimension

        self.beta = max(beta, 1)

        self.hyper_parameter_list = [self.number_of_epochs,
                                     self.batch_size,
                                     self.learning_rate,
                                     self.encoder_activation,
                                     self.decoder_activation,
                                     self.enable_batch_normalization,
                                     self.enable_dropout,
                                     self.dropout_rate,
                                     self.l2_constant,
                                     self.early_stopping_patience,
                                     self.early_stopping_delta,
                                     self.latent_dimension]

        if is_mnist:
            self.hyper_parameter_list.append("mnist")

        if is_restricted:
            restriction_string = ''
            for number in restriction_labels:
                restriction_string += str(number) + ','
            self.hyper_parameter_list.append(f"restricted_{restriction_string[:-1]}")

        if angle_threshold != 180:
            self.hyper_parameter_list.append((f"threshold_{angle_threshold}"))

        if enable_augmentation:
            augmentation_string = "_".join(["augmented", str(covariance_coefficient), str(augmentation_size)])
            self.hyper_parameter_list.append(augmentation_string)

        if not enable_activation:
            self.hyper_parameter_list.append("PCA")

        if enable_rotations:
            self.hyper_parameter_list.append(f"rotated_{number_of_rotations},{angle_of_rotation}")

        if beta > 1:
            self.hyper_parameter_list.append(f"beta_{beta}")

        self.hyper_parameter_list.append(f"alpha_{smoothing_alpha}")

        if is_confused:
            self.hyper_parameter_list.append("confused")

        self.hyper_parameter_string = '_'.join([str(i) for i in self.hyper_parameter_list])

        self.directory_counter = directories.DirectoryCounter(self.hyper_parameter_string)
        self.directory_number = self.directory_counter.count()
        self.hyper_parameter_string = '_'.join([self.hyper_parameter_string, 'x{:02d}'.format(self.directory_number)])

        directory, image_directory = directories.DirectoryCounter.get_output_directory(self.hyper_parameter_string,
                                                                                       self.model_name)
        self.experiment_directory = directory
        self.image_directory = image_directory
        if not self.predict:
            directories.DirectoryCounter.make_output_directory(self.hyper_parameter_string, self.model_name)

        """
        Tensorflow Input instances for declaring model inputs.
        """
        self.mnist_shape = self.x_train.shape[1:]
        self.gaussian_shape = 2 * self.latent_dimension
        self.encoder_gaussian = Input(shape=self.gaussian_shape, name='enc_gaussian')
        self.encoder_mnist_input = Input(shape=self.mnist_shape, name='enc_mnist')
        self.auto_encoder_gaussian = Input(shape=self.gaussian_shape, name='ae_gaussian')
        self.auto_encoder_mnist_input = Input(shape=self.mnist_shape, name='ae_mnist')

        """
        Callbacks.
        """
        self.tensorboard_callback = TensorBoard(log_dir=os.path.join(self.experiment_directory, 'tensorboard_logs'),
                                                histogram_freq=1,
                                                write_graph=False,
                                                write_images=True)

        self.early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                     min_delta=self.early_stopping_delta,
                                                     patience=self.early_stopping_patience,
                                                     mode='auto',
                                                     restore_best_weights=True)

        self.learning_rate_callback = ReduceLROnPlateau(monitor='val_loss',
                                                        factor=0.1,
                                                        patience=self.lr_reduction_patience,
                                                        min_lr=self.learning_rate_minimum)

        self.nan_termination_callback = TerminateOnNaN()

        self.colors = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']

    def print_settings(self):
        for t in self.__dict__.items():
            print(t)

    def define_encoder(self):
        """
        Abstract method for encoder instantiation.
        :return: encoder, encoder_output
        """
        return None, None

    def define_decoder(self, encoder_output):
        """
        Abstract method for decoder instantiation.
        :return: decoder
        """
        return None

    def define_autoencoder(self):
        """
        Abstract method for autoencoder instantiation.
        :return: autoencoder, encoder, decoder
        """
        return None, None, None

    def get_fit_args(self):
        """
        Define a list of NumPy inputs and NumPy outputs of the Keras model. These are the actual data that
        flow through the Keras model.
        :return: A list of arguments for the fit method of the Keras model.
        """
        return [[self.gaussian_train, self.x_train], [self.gaussian_train, self.x_train]]

    def get_fit_kwargs(self):
        """
        Construct keyword arguments for fitting the Keras model. This is useful for conditioning the model's training
        on the presence of a validation set.
        :return: A dictionary of keyword arguments for the fit method of the Keras model.
        """
        fit_kwargs = dict()
        fit_kwargs['epochs'] = self.number_of_epochs
        fit_kwargs['batch_size'] = self.batch_size
        if self.has_validation_set and self.enable_early_stopping:
            fit_kwargs['callbacks'] = [self.early_stopping_callback, self.nan_termination_callback]
        else:
            fit_kwargs['callbacks'] = [self.nan_termination_callback]
        if self.has_validation_set:
            fit_kwargs['validation_data'] = ([self.gaussian_val, self.x_val], [self.gaussian_val, self.x_val])
        return fit_kwargs

    def fit_autoencoder(self):
        """
        Fit the autoencoder to the data.
        :return: A 4-tuple consisting of the autoencoder, encoder, and decoder Keras models, along with the history of
            the autoencoder, which stores training and validation metrics.
        """
        args = self.get_fit_args()
        kwargs = self.get_fit_kwargs()
        auto_encoder, encoder, decoder = self.define_autoencoder()
        history = auto_encoder.fit(*args, **kwargs)
        return auto_encoder, encoder, decoder, history

    def assign_soft_labels(self, x_train_latent, x_test_latent):
        """
        Fit a Gaussian mixture model to a latent representation of training data, then use the components in the
        mixture model to return a collection of class probabilities for a latent representation of test data.
        :param x_train_latent: A NumPy data array.
        :param x_test_latent: A NumPy data array.
        :return: A NumPy data array of soft class probabilities.
        """
        mixture_model = GaussianMixture(self.number_of_clusters)
        mixture_model.fit(x_train_latent)
        # Get the parameters for each Gaussian density in the mixture model, then fire up SciPy to compute the density
        # values for each data point in the latent representation.
        means = mixture_model.means_
        covariances = mixture_model.covariances_
        soft_labels = GaussianSoftLabels(x_test_latent, means, covariances, labels=self.y_test_binary)
        return soft_labels.smooth_gaussian_mixture(alpha=self.alpha)

    def plot_results(self, models):
        """Plots labels and MNIST digits as a function of the 2D latent vector

        # Arguments
            models (tuple): encoder and decoder models
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """
        encoder, decoder = models
        test_gaussian = operations.get_gaussian_parameters(self.x_test, self.latent_dimension)
        os.makedirs(self.image_directory, exist_ok=True)

        filename = "vae_mean.png"
        filepath = os.path.join(self.image_directory, filename)

        z_gaussian, z_data = encoder.predict([test_gaussian, self.x_test], batch_size=self.batch_size)
        z_mean, z_covariance = operations.split_gaussian_parameters(z_gaussian)

        if self.latent_dimension == 2:
            # display a 2D plot of the data classes in the latent space
            plt.figure(figsize=(12, 10))
            plt.scatter(z_mean[:, 0], z_mean[:, 1], c=self.y_test, s=8, alpha=0.3)
            plt.colorbar(ticks=np.linspace(0, 2, 3))
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.savefig(filepath, dpi=200)
            if self.show:
                plt.show()
        else:
            # display a 2D t-SNE of the data classes in the latent space
            plt.figure(figsize=(12, 10))
            tsne = LatentSpaceTSNE(z_mean, self.y_test, self.experiment_directory)
            tsne.save_tsne()

        if self.latent_dimension == 2:
            if self.is_mnist:
                filename = "latent.png"
                filepath = os.path.join(self.image_directory, filename)
                # display a 30x30 2D manifold of digits
                n = 30
                image_size = 28
                figure = np.zeros((image_size * n, image_size * n))
                # linearly spaced coordinates corresponding to the 2D plot
                # of digit classes in the latent space
                grid_x = np.linspace(-4, 4, n)
                grid_y = np.linspace(-4.5, 3.5, n)[::-1]

                for i, yi in enumerate(grid_y):
                    for j, xi in enumerate(grid_x):
                        parameter_tuple = (np.zeros(self.latent_dimension), np.ones(self.latent_dimension))
                        dummy_gaussian = np.asarray([np.concatenate(parameter_tuple)])
                        z_sample = np.array([[xi, yi]])
                        x_decoded = decoder.predict([dummy_gaussian, z_sample])
                        digit = x_decoded[1].reshape(image_size, image_size)
                        figure[i * image_size: (i + 1) * image_size,
                        j * image_size: (j + 1) * image_size] = digit

                plt.figure(figsize=(10, 10))
                start_range = image_size // 2
                end_range = (n - 1) * image_size + start_range + 1
                pixel_range = np.arange(start_range, end_range, image_size)
                sample_range_x = np.round(grid_x, 1)
                sample_range_y = np.round(grid_y, 1)
                plt.xticks(pixel_range, sample_range_x)
                plt.yticks(pixel_range, sample_range_y)
                plt.xlabel("z[0]")
                plt.ylabel("z[1]")
                plt.imshow(figure, cmap='Greys_r')
                plt.savefig(filepath)
                if self.show:
                    plt.show()
                plt.close('all')

            else:
                filename = "latent.png"
                filepath = os.path.join(self.image_directory, filename)
                # display a latent representation
                n = 30
                image_size = 224
                figure = np.zeros((image_size * n, image_size * n))
                # linearly spaced coordinates corresponding to the 2D plot
                # of digit classes in the latent space
                grid_x = np.linspace(-4, 4, n)
                grid_y = np.linspace(-4.5, 3.5, n)[::-1]

                for i, yi in enumerate(grid_y):
                    for j, xi in enumerate(grid_x):
                        parameter_tuple = (np.zeros(self.latent_dimension), np.ones(self.latent_dimension))
                        dummy_gaussian = np.asarray([np.concatenate(parameter_tuple)])
                        z_sample = np.array([[xi, yi]])
                        x_decoded = decoder.predict([dummy_gaussian, z_sample])
                        digit = x_decoded[1].reshape(image_size, image_size)
                        figure[i * image_size: (i + 1) * image_size,
                        j * image_size: (j + 1) * image_size] = digit

                plt.figure(figsize=(10, 10))
                start_range = image_size // 2
                end_range = (n - 1) * image_size + start_range + 1
                pixel_range = np.arange(start_range, end_range, image_size)
                sample_range_x = np.round(grid_x, 1)
                sample_range_y = np.round(grid_y, 1)
                plt.xticks(pixel_range, sample_range_x)
                plt.yticks(pixel_range, sample_range_y)
                plt.xlabel("z[0]")
                plt.ylabel("z[1]")
                plt.imshow(figure, cmap='Greys_r')
                plt.savefig(filepath)
                if self.show:
                    plt.show()
                plt.close('all')

    def train(self):
        """
        Begin logging, train the autoencoder, use the autoencoder's history to plot loss curves, and save_all the parameters
        of the autoencoder, encoder, and decoder (respectively) to .h5 files.
        :return: None
        """
        t0 = time()
        if self.enable_logging:
            logs.begin_logging(self.experiment_directory)
        auto_encoder, encoder, decoder, history = self.fit_autoencoder()
        t1 = time()
        t = t1 - t0
        print(f"Variational autoencoder trained in {t} seconds.\n")

        plots.loss(history, self.image_directory)

        self.save_model_weights(auto_encoder, 'auto_encoder')
        self.save_model_weights(encoder, 'encoder')
        self.save_model_weights(decoder, 'decoder')

        #self.save_prediction(encoder, data_filename='x_train_latent.npy')
        self.save_prediction(encoder,
                             data=[self.gaussian_test, self.x_test],
                             data_filename='x_test_latent.npy')
        #self.save_prediction(auto_encoder, data_filename='x_train_predict.npy')
        self.save_prediction(auto_encoder,
                             data=[self.gaussian_test, self.x_test],
                             data_filename='x_test_predict.npy')
        self.save_experiment_settings()
        x_test_pred_file = os.path.abspath(os.path.join(self.experiment_directory, 'x_test_predict.npy'))
        x_test_pred = np.load(x_test_pred_file)
        self.save_input_output_comparison(self.x_test, x_test_pred)
        self.report_latent_space_classifiers(encoder)
        self.plot_results((encoder, decoder))

        return auto_encoder, encoder, decoder

    def load_model_weights(self, weight_directory, architecture='auto_encoder'):
        """
        Load the weights of the model specified by the class constructor.
        :return: None
        """
        architecture_set = {'auto_encoder', 'encoder', 'decoder'}
        assert architecture in architecture_set, "Invalid model name. Model names are: 'encoder', 'decoder', 'auto_encoder'."

        if architecture == 'encoder':
            model, _ = self.define_encoder()
        elif architecture == 'decoder':
            _, encoder_output = self.define_encoder()
            model = self.define_decoder(encoder_output)
        else:
            model, _, _ = self.define_autoencoder()

        filepath = os.path.abspath(
            os.path.join(self.experiment_directory, '..', weight_directory, 'models', architecture + '.h5'))
        model.load_weights(filepath)
        return model

    def save_model_weights(self, model, model_name):
        """
        Save the weights of a model to an .h5 file.
        :param model: A Keras model instance.
        :param model_name: A string indicating the model's filename.
        :return: None
        """
        model_directory = os.path.join(self.experiment_directory, 'models')
        model_filepath = os.path.join(model_directory, model_name + '.h5')
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model.save_weights(model_filepath)

    def save_prediction(self, model, data=None, labels=None, data_filename=None, labels_filename=None):
        """
        Save a model's predictions as NumPy arrays. If no data are specified, then the variational autoencoder's
        training set is given as input to the model.
        :param model: A Keras model, in this case either the full autoencoder or only the encoder.
        :param data: A list of two NumPy arrays (Gaussian parameters and images) for input to the model.
        :param labels: A NumPy array of labels for the input data.
        :return: None
        """
        if data is None:
            data = [self.gaussian_train, self.x_train]
        prediction = model.predict(data)
        prediction = prediction[1]

        if data_filename:
            file_path = os.path.abspath(os.path.join(self.experiment_directory, data_filename))
        else:
            file_path = os.path.abspath(os.path.join(self.experiment_directory, 'latent_data.npy'))
        np.save(file_path, prediction)

        if labels is not None:
            if labels_filename:
                file_path = os.path.abspath(os.path.join(self.experiment_directory, labels_filename))
            else:
                file_path = os.path.abspath(os.path.join(self.experiment_directory, 'latent_labels.npy'))
            np.save(file_path, labels)

    def get_prediction(self, model, data=None, latent_only=True):
        """
        Run a prediction on the given data set.
        :param model: A Keras model. In this case, either the autoencoder, the encoder, or the decoder.
        :param data: The data on which to get_prediction. Default is None. If None, then data is set to the training data.
        :return: The model's prediction of the data.
        """
        if data is None:
            data = [self.gaussian_train, self.x_train]
        prediction = model.predict(data)
        if latent_only:
            return prediction[1]
        else:
            return prediction

    def generate(self, decoder, number_of_samples=1):
        """
        Generate samples using the decoder of the learned autoencoder's generative model.
        :param decoder: A Keras model. Here's it's a decoder learned by training a VAE.
        :param number_of_samples: An integer denoting the number of samples to generate. Default is 1.
        :return: A NumPy array of data produced by the generative model.
        """
        # data = samples in the latent space.
        # return self.get_prediction(decoder, data)

    def get_mixture_model(self, model, data, labels):
        latent_representation = self.get_prediction(model, data=data, latent_only=True)
        if len(labels[0] > 1):
            labels = OneHotDecoder(labels).decode_to_multiclass()
        return classifiers.fit_mixture_model_on_latent_space(latent_representation, labels), latent_representation

    def get_logistic_regression(self, model, data, labels):
        latent_representation = self.get_prediction(model, data=data, latent_only=True)
        return classifiers.logistically_regress_on_latent_space(latent_representation, labels), latent_representation

    def get_support_vector_classification(self, model, data, labels):
        latent_representation = self.get_prediction(model, data=data, latent_only=True)
        if len(labels[0] > 1):
            labels = OneHotDecoder(labels).decode_to_multiclass()
        return classifiers.sv_classify_on_latent_space(latent_representation, labels, kernel='rbf'), \
               latent_representation

    def report_score(self, model, data, labels, text, file):
        if len(labels.shape) > 1:
            labels = OneHotDecoder(labels).decode_to_multiclass()
        score = model.score(data, labels)
        score_string = text + f" {score}\n"
        print(text, score)
        file.write(score_string)

    def report_latent_space_classifiers(self, encoder):
        filepath = os.path.abspath(os.path.join(self.experiment_directory, 'classifiers.log'))
        classifier_report = open(filepath, "w+")
        if self.with_logistic_regression:
            logistic_regression, latent_representation = self.get_logistic_regression(encoder,
                                                                                      [self.gaussian_test, self.x_test],
                                                                                      self.y_test)
            self.report_score(logistic_regression,
                              latent_representation,
                              self.y_test,
                              "Logistic regression model score:",
                              classifier_report)

        if self.with_mixture_model:
            mixture_model, latent_representation = self.get_mixture_model(encoder,
                                                                          [self.gaussian_test, self.x_test],
                                                                          self.y_test)
            self.report_score(mixture_model,
                              latent_representation,
                              self.y_test,
                              "Gaussian mixture model per-sample average log-likelihood:",
                              classifier_report)

        if self.with_svc:
            svc, latent_representation = self.get_support_vector_classification(encoder,
                                                                                [self.gaussian_test,
                                                                                 self.x_test],
                                                                                self.y_test)
            self.report_score(svc,
                              latent_representation,
                              self.y_test,
                              "Support vector classifier mean accuracy:",
                              classifier_report)
        classifier_report.close()

    def save_data(self):
        print('Saving data.')
        np.save(os.path.join(self.experiment_directory, 'x_train.npy'), self.x_train)
        np.save(os.path.join(self.experiment_directory, 'x_val.npy'), self.x_val)
        np.save(os.path.join(self.experiment_directory, 'x_test.npy'), self.x_test)
        np.save(os.path.join(self.experiment_directory, 'y_train.npy'), self.y_train)
        np.save(os.path.join(self.experiment_directory, 'y_val.npy'), self.y_val)
        np.save(os.path.join(self.experiment_directory, 'y_test.npy'), self.y_test)
        if self.is_confused:
            np.save(os.path.join(self.experiment_directory, 'a_train.npy'), self.a_train)
            np.save(os.path.join(self.experiment_directory, 'a_val.npy'), self.a_val)
            np.save(os.path.join(self.experiment_directory, 'a_test.npy'), self.a_test)

    def get_experiment_dictionary(self):
        """
        Return the class dictionary, absent non-compatible types.
        :return: A dictionary of settings for the experiment.
        """
        return {key: self.__dict__[key]
                for key in self.__dict__
                if isinstance(self.__dict__[key], bool)
                or isinstance(self.__dict__[key], int)
                or isinstance(self.__dict__[key], float)
                or isinstance(self.__dict__[key], str)
                or isinstance(self.__dict__[key], list)}

    def save_experiment_settings(self):
        filename = os.path.abspath(os.path.join(self.experiment_directory, 'experiment.json'))
        experiment_dictionary = self.get_experiment_dictionary()
        with open(filename, 'w') as f:
            json.dump(experiment_dictionary, f, indent=4)

    def save_input_output_comparison(self,
                                     input,
                                     output,
                                     idx=0,
                                     number_of_rows=2,
                                     number_of_columns=5,
                                     figure_scale=8):
        if not self.is_mnist:
            # This is a hack for the x-ray data set. This condition should be removed.
            input = np.reshape(input, input.shape[0: -1])
            output = np.reshape(output, output.shape[0: -1])

        input_array_list = [input[idx + i] for i in range(number_of_columns)]
        output_array_list = [output[idx + i] for i in range(number_of_columns)]
        array_list = input_array_list + output_array_list

        figure_size_tuple = (figure_scale, figure_scale * number_of_rows / number_of_columns)

        fig = plt.figure(figsize=figure_size_tuple)
        for i in range(len(array_list)):
            img = array_list[i]
            fig.add_subplot(number_of_rows, number_of_columns, i + 1)
            plt.imshow(img)
        filename = os.path.abspath(os.path.join(self.image_directory,
                                                f'reconstruction_{idx}:{idx + number_of_columns - 1}.png'))
        plt.savefig(filename)
        del fig
