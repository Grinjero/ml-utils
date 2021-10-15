import tensorflow as tf
from tensorflow.keras import layers


class ArBlockLSTM(tf.keras.Model):
    def __init__(self, out_steps, block_size, units, num_out_features=1):
        super(ArBlockLSTM, self).__init__()

        self.out_steps = out_steps
        self.block_size = block_size

        assert (out_steps % block_size) == 0

        self.units = units
        self.num_out_features = num_out_features
        self._construct()

    def _construct(self):
        self.lstm_cell = layers.LSTMCell(self.units, name="LSTM_cell")
        self.rnn = layers.RNN(self.lstm_cell, return_state=True, name="RNN_wrapper")
        self.dense = layers.Dense(self.num_out_features * self.block_size, name="out_dense")

    def _warmup(self, inputs):
        x, *state = self.rnn(inputs)
        # prediction for the last input

        prediction = self.dense(x)
        return prediction, state

    def forecast(self, inputs, num_steps, training):
        assert (num_steps % self.block_size) == 0
        prediction, state = self._warmup(inputs)

        batch_size = tf.shape(inputs)[0]
        predictions = tf.zeros((batch_size, num_steps, self.num_out_features))
        predictions[:, 0, :] = prediction

        for i in range(1, num_steps):
            prediction, state = self.lstm_cell(prediction, state=state, training=training)
            prediction = self.dense(prediction)
            # prediction_dim (batch_size, num_out_features * block_size)
            # reshape (batch_size, num_out_features * block_size) -> (batch_size, block_size, num_out_features)

            prediction = tf.reshape(prediction, (batch_size, self.block_size, self.num_out_features))
            predictions[:, i * self.block_size:(i + 1) * self.block_size, :] = prediction

        return predictions

    def call(self, inputs, training=False, mask=None):
        return self.forecast(inputs, num_steps=self.out_steps, training=training)
