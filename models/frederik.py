import keras as ks

n_blocks = 2
filters = 8


def convblock(tns, filters, kernel_size=3, padding='same'):
    tns = ks.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(tns)
    tns = ks.layers.BatchNormalization()(tns)
    tns = ks.layers.LeakyReLU()(tns)
    tns = ks.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(tns)
    tns = ks.layers.BatchNormalization()(tns)
    tns = ks.layers.LeakyReLU()(tns)
    tns = ks.layers.MaxPooling2D(2)(tns)
    return tns


inp = ks.layers.Input(shape=(28, 28, 1))
tns = inp

for _ in range(n_blocks):
    tns = convblock(tns, filters)
    filters *= 2

tns = ks.layers.Flatten()(tns)
out = ks.layers.Dense(1, activation='sigmoid')(tns)

model = ks.models.Model(inputs=inp, outputs=out)
model.save('model.h5')

model.summary()
