"""Test pretrained NALU model"""

from math import floor
import numpy as np
from keras_nalu.pretrained.model import get_B, get_model

def encode(num, b):
    """Encode decimal number as vector of digits"""

    a = np.zeros((len(b)))
    b_whole = np.floor(b / b[-1])
    num_whole = floor(num / b[-1])

    for i in range(len(b)):
        a[i] = floor(num_whole / b_whole[i]) % 10

    return np.concatenate([a, b])

def generate_numbers():
    """Generate common decimal numbers"""

    nums = []

    for i in range(0, 12):
        for j in range(1, 12):
            nums.append(i / j)

    return list(set(nums))

def test_pretrained():
    """Test loss of pretrained model on common decimal numbers"""

    model = get_model()
    nums = generate_numbers()
    b = get_B(base=10, precision=[7, -9], size=1)[0]
    X = np.zeros((len(nums), 2 * len(b)))
    Y = np.zeros((len(nums), 1))

    for i, num in enumerate(nums):
        X[i] = encode(num, b)
        Y[i][0] = num

    loss = model.evaluate(x=X, y=Y)

    assert loss < 1e-5
