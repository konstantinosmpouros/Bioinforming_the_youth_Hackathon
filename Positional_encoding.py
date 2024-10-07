import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angle_rates
    
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)  # Ensure inputs are float32
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]