"""Keras NALU pretrained model"""

from os import path
from keras.models import load_model

def get_model():
    """Get the NALU pretrained model"""

    return load_model(
        path.join(path.dirname(__file__), 'model.h5')
    )
