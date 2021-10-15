import tensorflow as tf
from tensorflow.keras import layers


class ArLSTM(tf.keras.Model):
    def __init__(self, out_steps, units, num_out_features=1):
        super(ArLSTM, self).__init__()

        self.out_steps = out_steps
        self.units = units
        self.num_out_features = num_out_features
        self._construct()

    def _construct(self):
        self.lstm_cell = layers.LSTMCell(self.units, name="LSTM_cell")
        self.rnn = layers.RNN(self.lstm_cell, return_state=True, name="RNN_wrapper")
        self.dense = layers.Dense(self.num_out_features, name="out_dense")

    def _warmup(self, inputs):
        x, *state = self.rnn(inputs)
        # prediction for the last input

        prediction = self.dense(x)
        return prediction, state

    def forecast(self, inputs, num_steps, training):
        prediction, state = self._warmup(inputs)

        predictions = [prediction]

        for i in range(1, num_steps):
            prediction, state = self.lstm_cell(prediction, states=state, training=training)
            prediction = self.dense(prediction)

            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def call(self, inputs, training=False, mask=None):
        return self.forecast(inputs, num_steps=self.out_steps, training=training)
