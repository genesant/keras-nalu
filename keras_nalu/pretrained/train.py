"""Pretrain Keras NALU model on counting task"""

from os import path
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
from keras_nalu.nalu import NALU

def generate_dataset(batch_size, number_width, multiplier):
    """Generate dataset for a task"""

    X = np.zeros((batch_size, 2 * number_width))
    Y = np.zeros((batch_size))

    for i in range(batch_size):
        a = multiplier * np.random.rand(number_width)
        b = multiplier * np.random.rand(number_width)
        X[i] = np.concatenate([a, b])
        Y[i] = np.sum(a * b)

    return X, Y

def train():
    """Train Keras NALU model on counting task"""

    model_dir = path.dirname(__file__)
    number_width = 16

    X_train, Y_train = generate_dataset(
        batch_size=2**18,
        multiplier=1,
        number_width=number_width,
    )

    X_validation, Y_validation = generate_dataset(
        batch_size=2**9,
        multiplier=9999,
        number_width=number_width,
    )

    X_test, Y_test = generate_dataset(
        batch_size=2**9,
        multiplier=9999,
        number_width=number_width,
    )

    inputs = Input(shape=(2 * number_width,))
    hidden = NALU(units=number_width, cell='m')(inputs)
    outputs = NALU(units=1, cell='a')(hidden)

    callbacks = [
        TerminateOnNaN(),
        ReduceLROnPlateau(
            factor=0.1,
            min_lr=1e-16,
            patience=50,
            verbose=1,
        ),
        EarlyStopping(
            patience=200,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(loss='mae', optimizer=RMSprop(lr=0.01))

    model.fit(
        batch_size=256,
        callbacks=callbacks,
        epochs=1000,
        validation_data=(X_validation, Y_validation),
        verbose=2,
        x=X_train,
        y=Y_train,
    )

    model.evaluate(
        batch_size=256,
        verbose=1,
        x=X_test,
        y=Y_test,
    )

    model.save(path.join(model_dir, 'model.h5'))

if __name__ == '__main__':
    train()
