import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

class Umbrella:
    def __init(self, input_shape=(224, 224, 3)):
        
        inputs = tf
        self.x = Input(shape=input_shape)
        self.deploy()
        self.model = Model(inputs, x)
        