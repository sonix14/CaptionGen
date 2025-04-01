import tensorflow as tf
from attention_model import Attention

class Decoder(tf.keras.Model):
  def __init__(self, enc_dimension, units, vocab_size):
    super(Decoder, self).__init__()
    self.units = units
    self.embedding = tf.keras.layers.Embedding(
      vocab_size, enc_dimension)
    self.gru = tf.keras.layers.GRU(
      self.units,
      return_sequences=True,
      return_state=True,
      recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.attention = Attention(self.units)

  def call(self, x, features, hidden):
    # Получаем вектор контекста и веса
    context_vector, attention_weights = self.attention(
      features, hidden)

    # Приводим форму предыдущего выхода к форме вектора контекста
    x = self.embedding(x)

    # Дополняем вектор контекста предыдущим выходом декодировщика
    # Размерность x: (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # GRU подбирает токен
    x, state = self.gru(x)
    x = self.fc1(x)
    # Сокращаем размерность: (batch_size * max_token_len, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))
    # Меняем размерность на (batch_size * max_token_len, vocab_size)
    # Т.е. происходит активация не всех выходов предыдущего слоя
    # Т.к. нам нужна размерность vocab_size, а не hidden
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
