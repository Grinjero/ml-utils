import tensorflow as tf
from tensorflow.keras import layers, models


class Seq2SeqLSTM(tf.keras.Model):
    def __init__(self, input_size, output_size, out_steps, hidden_units, num_stacked_rnn):
        super(Seq2SeqLSTM, self).__init__()

        self.out_steps = out_steps
        self.num_stacked_rnn = num_stacked_rnn
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units

        self._construct_encoder()
        self._construct_decoder()

    def _construct_encoder(self):
        self.encoder_layers = []

        for i in range(self.num_stacked_rnn):
            ret_seq = i != (self.num_stacked_rnn - 1)
            layer = layers.LSTM(self.hidden_units, return_sequences=ret_seq, return_state=True)
            self.encoder_layers.append(layer)

    def _construct_decoder(self):
        self.decoder_layers = []

        for i in range(self.num_stacked_rnn):
            cell = layers.LSTM(self.hidden_units, return_sequences=True)
            self.decoder_layers.append(cell)

        self.context_repeater = layers.RepeatVector(self.out_steps)
        self.out_fc = layers.Dense(self.output_size)
        self.out = layers.TimeDistributed(self.out_fc)

    def call(self, inputs, training=None, mask=None):
        previous_outs = inputs

        states = []
        for encoder_layer in self.encoder_layers:
            previous_outs, *state = encoder_layer(previous_outs)
            states.append(state)

        # context vector
        previous_outs = self.context_repeater(previous_outs)
        for decoder_layer, encoder_state in zip(self.decoder_layers, states):
            previous_outs = decoder_layer(previous_outs, initial_state=encoder_state)

        forecast = self.out(previous_outs)
        return forecast

