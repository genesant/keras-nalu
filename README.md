# Keras NALU (Neural Arithmetic Logic Units)

[![CircleCI](https://circleci.com/gh/genesant/keras-nalu/tree/master.svg?style=shield)](https://circleci.com/gh/genesant/keras-nalu/tree/master)

Keras implementation of a NALU layer (Neural Arithmetic Logic Units).
See: https://arxiv.org/pdf/1808.00508.pdf.

## Installation

```
pip install keras-nalu
```

## Usage

```py
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras_nalu.nalu import NALU

# Your dataset
X_test = ... # Interpolation data
Y_test = ... # Interpolation data

X_validation = ... # Extrapolation data (validation)
Y_validation = ... # Extrapolation data (validation)

X_test = ... # Extrapolation data (test)
Y_test = ... # Extrapolation data (test)

# Hyper parameters
epoch_count=1000
learning_rate = 0.05
seq_len = 100

inputs = Input(shape=(seq_len, ))
hidden = NALU(units=2)(inputs)
hidden = NALU(units=2)(hidden)
outputs = NALU(units=1)(hidden)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))

model.fit(
    batch_size=256,
    epochs=epoch_count,
    validation_data=(X_validation, Y_validation),
    x=X_train,
    y=Y_train,
)

extrapolation_loss = model.evaluate(
    batch_size=256,
    x=X_test,
    y=Y_test,
)
```
