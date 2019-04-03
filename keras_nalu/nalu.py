"""Keras NALU module"""

from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import InputSpec
from keras.layers import Layer
from keras.utils.generic_utils import get_custom_objects

class NALU(Layer):
    """Keras NALU layer"""

    def __init__(
            self,
            units,
            G_constraint=None,
            G_initializer='glorot_uniform',
            G_regularizer=None,
            M_hat_constraint=None,
            M_hat_initializer='glorot_uniform',
            M_hat_regularizer=None,
            W_hat_constraint=None,
            W_hat_initializer='glorot_uniform',
            W_hat_regularizer=None,
            e=1e-7,
            **kwargs,
    ):
        super(NALU, self).__init__(**kwargs)
        self.G = None
        self.G_constraint = constraints.get(G_constraint)
        self.G_initializer = initializers.get(G_initializer)
        self.G_regularizer = regularizers.get(G_regularizer)
        self.M_hat = None
        self.M_hat_constraint = constraints.get(M_hat_constraint)
        self.M_hat_initializer = initializers.get(M_hat_initializer)
        self.M_hat_regularizer = regularizers.get(M_hat_regularizer)
        self.W_hat = None
        self.W_hat_constraint = constraints.get(W_hat_constraint)
        self.W_hat_initializer = initializers.get(W_hat_initializer)
        self.W_hat_regularizer = regularizers.get(W_hat_regularizer)
        self.e = e
        self.supports_masking = True
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.G = self.add_weight(
            constraint=self.G_constraint,
            initializer=self.G_initializer,
            name='G',
            regularizer=self.G_regularizer,
            shape=(input_dim, self.units),
        )

        self.M_hat = self.add_weight(
            constraint=self.M_hat_constraint,
            initializer=self.M_hat_initializer,
            name='M_hat',
            regularizer=self.M_hat_regularizer,
            shape=(input_dim, self.units),
        )

        self.W_hat = self.add_weight(
            constraint=self.W_hat_constraint,
            initializer=self.W_hat_initializer,
            name='W_hat',
            regularizer=self.W_hat_regularizer,
            shape=(input_dim, self.units),
        )

        self.built = True
        self.input_spec = InputSpec(axes={-1: input_dim}, min_ndim=2)

    def call(self, inputs, **kwargs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat)
        a = K.dot(inputs, W)
        m = K.exp(K.dot(K.log(K.abs(inputs) + self.e), W))
        g = K.sigmoid(K.dot(inputs, self.G))
        y = (g * a) + ((1 - g) * m)
        return y

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)
        return output_shape

    def get_config(self):
        base_config = super(NALU, self).get_config()

        config = {
            'G_constraint': constraints.serialize(self.G_constraint),
            'G_initializer': initializers.serialize(self.G_initializer),
            'G_regularizer': regularizers.serialize(self.G_regularizer),
            'M_hat_constraint': constraints.serialize(self.M_hat_constraint),
            'M_hat_initializer': initializers.serialize(self.M_hat_initializer),
            'M_hat_regularizer': regularizers.serialize(self.M_hat_regularizer),
            'W_hat_constraint': constraints.serialize(self.W_hat_constraint),
            'W_hat_initializer': initializers.serialize(self.W_hat_initializer),
            'W_hat_regularizer': regularizers.serialize(self.W_hat_regularizer),
            'e': self.e,
            'units': self.units,
        }

        return {**base_config, **config}

get_custom_objects().update({'NALU': NALU})
