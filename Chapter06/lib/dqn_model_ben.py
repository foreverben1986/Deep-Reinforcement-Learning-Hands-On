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
        x = keras.layers.Conv2D(32, 
                                kernel_size = 8,
                                strides=(4, 4),
                                padding='same',
                                name="layer1", 
                                kernel_initializer="random_uniform", 
#                                 kernel_initializer="zeros", 
                                bias_initializer="zeros")(input_tensor)
#         x = tf.keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(64,
                                kernel_size = 4,
                                strides = (2, 2),
                                padding = "same",
                                name = "layer2",
                                kernel_initializer="random_uniform",
#                                 kernel_initializer="zeros",
                                bias_initializer="zeros")(x)
#         x = tf.keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(64,
                                kernel_size = 3,
                                strides = (1,1),
                                padding = "same",
                                name = "layer3",
                                kernel_initializer="random_uniform",
#                                 kernel_initializer="zeros",
                                bias_initializer="zeros")(x)
#         x = tf.keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(512, 
                               name="dense_layer1", 
                               kernel_initializer="random_uniform", 
                               bias_initializer="zeros")(x)
#         x = tf.keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dense(output, 
                               name="dense_layer2", 
                               kernel_initializer="random_uniform", 
                               bias_initializer="zeros")(x)
        model = tf.keras.Model(inputs = input_tensor, outputs = x)
        model.summary()
        return model
    
    def __compile__(self):
        self.keras_model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='mse',
              metrics=['accuracy'])
    def train(self, data, label, batch_size, epochs):
        self.keras_model.fit(data, label, batch_size=batch_size, epochs=epochs, verbose=0)
    def predict(self, data, batch_size):
        return self.keras_model.predict(data, batch_size=batch_size)
    def get_weights(self):
        return self.keras_model.get_weights()
    def save_weights(self, episodes):
        self.keras_model.save_weights("PongModelWeights" + str(episodes) + ".h5",overwrite=True)
    def load_weights(self, file):
        self.keras_model.load_weights(file)
    def export_model(self):
        config = self.keras_model.get_config()
        weights = self.keras_model.get_weights()

        new_model = keras.Model.from_config(config)
        new_model.set_weights(weights)
        return new_model
#     def export_saved_model(self, name):
#         keras.experimental.export_saved_model(self.keras_model, name)