"""Test Keras NALU module"""

from math import inf, isnan
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from src.nalu import NALU

SEQ_LEN = 100
INDICES = np.random.permutation(SEQ_LEN)
A_INDICES, B_INDICES = np.split(INDICES, 2) # pylint: disable=unbalanced-tuple-unpacking

def generate_dataset(size, task):
    """Generate dataset for a task"""

    X_train = np.random.rand(size, SEQ_LEN)
    X_validation = 2 * np.random.rand(size // 4, SEQ_LEN)
    X_test = 2 * np.random.rand(size, SEQ_LEN)

    a_train = np.sum(X_train[:, A_INDICES], axis=-1, keepdims=True)
    a_validation = np.sum(X_validation[:, A_INDICES], axis=-1, keepdims=True)
    a_test = np.sum(X_test[:, A_INDICES], axis=-1, keepdims=True)

    b_train = np.sum(X_train[:, B_INDICES], axis=-1, keepdims=True)
    b_validation = np.sum(X_validation[:, B_INDICES], axis=-1, keepdims=True)
    b_test = np.sum(X_test[:, B_INDICES], axis=-1, keepdims=True)

    Y_train = task(a_train, b_train)
    Y_validation = task(a_validation, b_validation)
    Y_test = task(a_test, b_test)

    return {
        'X_test': X_test,
        'X_train': X_train,
        'X_validation': X_validation,
        'Y_test': Y_test,
        'Y_train': Y_train,
        'Y_validation': Y_validation,
    }

def train(epoch_count, learning_rate, task):
    """Train a network on a task"""

    dataset = generate_dataset(size=2048, task=task)

    inputs = Input(shape=(SEQ_LEN, ))
    hidden = NALU(units=2)(inputs)
    hidden = NALU(units=2)(hidden)
    outputs = NALU(units=1)(hidden)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))

    result = model.fit(
        batch_size=256,
        callbacks=[EarlyStopping(patience=epoch_count, restore_best_weights=True)],
        epochs=epoch_count,
        verbose=2,
        validation_data=(dataset['X_validation'], dataset['Y_validation']),
        x=dataset['X_train'],
        y=dataset['Y_train'],
    )

    extrapolation_loss = model.evaluate(
        batch_size=256,
        verbose=1,
        x=dataset['X_test'],
        y=dataset['Y_test'],
    )

    interpolation_loss = result.history['loss'][-1]

    return model, interpolation_loss, extrapolation_loss

def train_retry(
        epoch_count,
        expected_extrapolation_loss,
        expected_interpolation_loss,
        learning_rate,
        task
):
    """Repeatedly train a network on a task until it converges"""

    model = None
    interpolation_loss = inf
    extrapolation_loss = inf

    for _ in range(5):
        model, interpolation_loss, extrapolation_loss = train(
            epoch_count=epoch_count,
            learning_rate=learning_rate,
            task=task,
        )

        if isnan(interpolation_loss):
            continue

        if isnan(extrapolation_loss):
            continue

        if interpolation_loss > expected_interpolation_loss:
            continue

        if extrapolation_loss > expected_extrapolation_loss:
            continue

        break

    assert interpolation_loss <= expected_interpolation_loss
    assert extrapolation_loss <= expected_extrapolation_loss
    return model, extrapolation_loss, extrapolation_loss
