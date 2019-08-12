import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class MyModel():
#     def __init__(self, name):
#         self.keras_model = keras.experimental.load_from_saved_model(name)
#         self.__compile__()
    def __init__(self, input_shape, outputs):
        inputs = tf.keras.Input(shape=input_shape)
        self.keras_model = self.build(inputs, outputs)
        self.__compile__()
    
    def build(self, input_tensor, output):
        x = keras.layers.Dense(128, name="layer1", kernel_initializer="random_uniform", bias_initializer="zeros")(input_tensor)
        x = keras.layers.Activation("relu", name="layer1_activation")(x)
        x = keras.layers.Dense(2, kernel_initializer="random_uniform", bias_initializer="zeros")(x)
        model = tf.keras.Model(inputs = input_tensor, outputs = x)
        model.summary()
        return model
    
    def __compile__(self):
        self.keras_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])
    def train(self, data, label, batch_size, epochs):
        self.keras_model.fit(data, label, batch_size=batch_size, epochs=epochs)
    def predict(self, data, batch_size):
        return self.keras_model.predict(data, batch_size=batch_size)
    def get_weights(self):
        return self.keras_model.get_weights()
    def export_model(self):
        config = self.keras_model.get_config()
        weights = self.keras_model.get_weights()

        new_model = keras.Model.from_config(config)
        new_model.set_weights(weights)
        return new_model
#     def export_saved_model(self, name):
#         keras.experimental.export_saved_model(self.keras_model, name)