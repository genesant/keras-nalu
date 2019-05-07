"""Keras NALU pretrained model"""

from os import path
from keras.models import load_model
import numpy as np

def get_B(base, precision, size):
    """Get B tensor of X coefficients"""

    exps = range(precision[0], precision[1], -1)
    pows = [base ** exp for exp in exps]
    B = np.tile(pows, (size, 1))
    return B

def get_model():
    """Get the NALU pretrained model"""

    return load_model(
        path.join(path.dirname(__file__), 'model.h5')
    )
