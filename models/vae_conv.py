from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow

import tensorflow.keras.backend as k
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from models.layers.vae_layers import Reparametrization
from models.losses.losses import EncodingLoss
from models.vae import VAE
from utils import logs, plots


class ConvolutionalVAE(VAE):
    def __init__(self,
                 deep=True,
                 enable_activation=True,
                 enable_augmentation=False,
                 enable_batch_normalization=True,
                 enable_dropout=True,
                 enable_early_stopping=False,
                 early_stopping_patience=20,
                 enable_logging=True,
                 enable_rotations=False,
                 enable_stochastic_gradient_descent=False,
                 has_custom_layers=True,
                 has_validation_set=True,
                 is_mnist=True,
                 is_restricted=False,
                 is_standardized=False,
                 with_logistic_regression=False,
                 show=False,
                 number_of_clusters=3,
                 restriction_labels=[1, 2, 3],
                 intermediate_dimension=512,
                 exponent_of_latent_space_dimension=1,
                 augmentation_size=100,
                 covariance_coefficient=0.2,
                 latent_dimension=2,
                 number_of_epochs=5,
                 batch_size=128,
                 learning_rate_initial=1e-5,
                 learning_rate_minimum=1e-6,
                 lr_reduction_patience=10,
                 dropout_rate=0.5,
                 l2_constant=1e-4,
                 early_stopping_delta=1,
                 beta=1,
                 smoothing_alpha=0.5,
                 number_of_rotations=2,
                 angle_of_rotation=30,
                 encoder_activation='relu',
                 decoder_activation='relu',
                 final_activation='sigmoid',
                 depth=5,
                 channel_size=8
                 ):
        model_name = 'vae_conv'
        self.depth = depth
        self.channel_size = channel_size
        super(ConvolutionalVAE, self).__init__(deep=deep,
                                               enable_activation=enable_activation,
                                               enable_augmentation=enable_augmentation,
                                               enable_batch_normalization=enable_batch_normalization,
                                               enable_dropout=enable_dropout,
                                               enable_early_stopping=enable_early_stopping,
                                               enable_logging=enable_logging,
                                               enable_rotations=enable_rotations,
                                               enable_stochastic_gradient_descent=enable_stochastic_gradient_descent,
                                               has_custom_layers=has_custom_layers,
                                               has_validation_set=has_validation_set,
                                               is_mnist=is_mnist,
                                               is_restricted=is_restricted,
                                               is_standardized=is_standardized,
                                               show=show,
                                               latent_dimension=latent_dimension,
                                               lr_reduction_patience=lr_reduction_patience,
                                               early_stopping_patience=early_stopping_patience,
                                               number_of_clusters=number_of_clusters,
                                               restriction_labels=restriction_labels,
                                               intermediate_dimension=intermediate_dimension,
                                               exponent_of_latent_space_dimension=exponent_of_latent_space_dimension,
                                               augmentation_size=augmentation_size,
                                               covariance_coefficient=covariance_coefficient,
                                               number_of_epochs=number_of_epochs,
                                               batch_size=batch_size,
                                               learning_rate_initial=learning_rate_initial,
                                               learning_rate_minimum=learning_rate_minimum,
                                               dropout_rate=dropout_rate,
                                               l2_constant=l2_constant,
                                               early_stopping_delta=early_stopping_delta,
                                               beta=beta,
                                               smoothing_alpha=smoothing_alpha,
                                               number_of_rotations=number_of_rotations,
                                               angle_of_rotation=angle_of_rotation,
                                               encoder_activation=encoder_activation,
                                               decoder_activation=decoder_activation,
                                               final_activation=final_activation,
                                               model_name=model_name,
                                               with_logistic_regression=with_logistic_regression)

    def conv_block(self, z, number_of_filters):

        z = Conv2D(filters=number_of_filters,
                   kernel_size=(3, 3),
                   padding='same',
                   activation=self.encoder_activation)(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)

        z = Conv2D(filters=number_of_filters,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation=self.encoder_activation)(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)
        return z

    def deconv_block(self, z, number_of_filters):
        z = Conv2DTranspose(filters=number_of_filters,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            activation=self.encoder_activation)(z)
        if self.enable_batch_normalization:
            z = BatchNormalization()(z)
        if self.enable_dropout:
            z = Dropout(rate=self.dropout_rate, seed=17)(z)
        return z

    def define_encoder(self):
        if self.is_mnist:
            input_data_tensor = self.encoder_mnist_input
            z = Reshape((28, 28, 1))(input_data_tensor)
        else:
            input_data_tensor = Input(shape=self.x_train.shape[1:], name='enc_input_tensor')
            z = input_data_tensor

        input_gaussian_tensor = self.encoder_gaussian

        for i in range(0, self.depth):
            number_of_filters = self.channel_size*(2**i)
            z = self.conv_block(z, number_of_filters)

        z = Flatten()(z)
        z_gaussian = Dense(self.gaussian_dimension, name="gaussian")(z)
        z = Reparametrization(name="latent_samples")(z_gaussian)
        encoder_output = [z_gaussian, z]

        encoder = Model([input_gaussian_tensor, input_data_tensor], encoder_output, name='encoder')
        encoder.summary()
        plot_model(encoder, to_file=os.path.join(self.image_directory, 'encoder.png'), show_shapes=True)

        return encoder, [z_gaussian, z]

    def define_decoder(self, encoder_output):
        """
        The decoder takes in two input tensors. The first is decoder_gaussian_input, which corresponds to the
        parameters of Gaussian distributions on latent space. The second is decoder_latent_input, which corresponds
        to latent space samples obtained from the learned Gaussians via the reparametrization trick.
        Since decoder_latent_input is a flattened vector, it needs to be reshaped for the decoder's convolution
        operations. The parameters for reshaping depend upon the number of convolution operations performed by the
        encoder, hence the arithmetic to obtain latent_image_dimension, latent_channel_size, and convolution_dimension.
        :param encoder_output: A Keras tensor, the output of the encoder model.
        :return: A Keras model, the decoder model.
        """
        decoder_gaussian_input = Input(shape=encoder_output[0].shape[1:], name='gaussian_input')
        decoder_latent_input = Input(shape=encoder_output[1].shape[1:], name='latent_input')
        x = decoder_latent_input
        gaussian = decoder_gaussian_input

        image_dimension = self.x_train.shape[1]
        latent_image_dimension = image_dimension/(2**self.depth)
        latent_image_dimension = int(latent_image_dimension)
        assert latent_image_dimension > 0, "Network depth is too large. Reduce --depth."

        latent_channel_size = self.channel_size * (2**(self.depth - 1))
        convolution_dimension = (latent_image_dimension ** 2) * latent_channel_size
        convolution_dimension = int(convolution_dimension)

        # Needed to prevent Keras from complaining that nothing was done to this tensor:
        identity_lambda = Lambda(lambda w: w, name="dec_identity_lambda")
        gaussian = identity_lambda(gaussian)

        x = Dense(convolution_dimension, activation=self.decoder_activation)(x)
        x = Reshape((latent_image_dimension, latent_image_dimension, latent_channel_size))(x)

        for i in range(self.depth - 2, -1, -1):
            number_of_filters = self.channel_size*(2**i)
            x = self.deconv_block(x, number_of_filters)

        x = self.deconv_block(x, 1)
        if self.is_mnist:
            x = Reshape((28, 28))(x)

        decoder_output = [gaussian, x]
        decoder = Model([decoder_gaussian_input, decoder_latent_input], decoder_output, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file=os.path.join(self.image_directory, 'decoder.png'), show_shapes=True)

        return decoder

    def define_autoencoder(self):
        encoder, z = self.define_encoder()
        decoder = self.define_decoder(z)

        if self.is_mnist:
            auto_encoder_input = [self.auto_encoder_gaussian, self.auto_encoder_mnist_input]
        else:
            auto_encoder_input_tensor = Input(shape=self.x_train.shape[1:], name='ae_input_tensor')
            auto_encoder_gaussian_tensor = Input(shape=self.gaussian_dimension, name="ae_gaussian_tensor")
            auto_encoder_input = [auto_encoder_gaussian_tensor, auto_encoder_input_tensor]
        latent_space_input = encoder(auto_encoder_input)
        auto_encoder_output = decoder(latent_space_input)
        auto_encoder = Model(auto_encoder_input, auto_encoder_output, name='variational_auto_encoder')
        encoding_loss = EncodingLoss()
        reconstruction_loss = tensorflow.keras.losses.MeanSquaredError()
        auto_encoder.summary()
        plot_model(auto_encoder, to_file=os.path.join(self.image_directory, 'auto_encoder.png'), show_shapes=True)
        auto_encoder.compile(optimizers.Adam(lr=self.learning_rate),
                             loss=[encoding_loss, reconstruction_loss],
                             loss_weights=[self.beta, self.data_dimension])
        return auto_encoder, encoder, decoder

    def get_fit_args(self):
        """
        Define a list of NumPy inputs and NumPy outputs of the Keras model. These are the actual data that flow through
        the Keras model.
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
            fit_kwargs['validation_data'] = ([self.gaussian_test, self.x_val], [self.gaussian_test, self.x_val])
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
        print("Variational autoencoder trained.\n")
        return auto_encoder, encoder, decoder, history

    def generate(self, decoder, number_of_samples=1):
        """
        Generate samples using the decoder of the learned autoencoder's generative model.
        :param decoder: A Keras model. Here's it's a decoder learned by training a VAE.
        :param number_of_samples: An integer denoting the number of samples to generate. Default is 1.
        :return: A NumPy array of data produced by the generative model.
        """
        # data = samples in the latent space.
        # return self.get_prediction(decoder, data)
