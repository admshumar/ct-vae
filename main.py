from models.vae_conv import ConvolutionalVAE

vae = ConvolutionalVAE(is_mnist=False,
                       has_validation_set=True,
                       number_of_epochs=1,
                       enable_early_stopping=True,
                       enable_logging=False,
                       enable_stochastic_gradient_descent=True,
                       encoder_activation='relu',
                       decoder_activation='relu',
                       final_activation='sigmoid',
                       learning_rate_initial=1e-4,
                       beta=1,
                       batch_size=4,
                       show=True)
vae.train()
del vae