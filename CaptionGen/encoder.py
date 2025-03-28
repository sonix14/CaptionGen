import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, dimension):
        super(Encoder, self).__init__()
        # Полносвязный слой
        self.layer = tf.keras.layers.Dense(dimension)

    def call(self, features):
        features = self.layer(features)
        # Заменяем отрицательные значения нулями
        features = tf.nn.relu(features)
        return features
