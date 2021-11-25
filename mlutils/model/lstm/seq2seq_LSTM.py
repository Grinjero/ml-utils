import tensorflow as tf
from tensorflow.keras import layers, models, Input


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
            layer = layers.LSTM(self.hidden_units, return_sequences=ret_seq, return_state=True, name=f"encoder_LSTM_{i}")
            self.encoder_layers.append(layer)

    def _construct_decoder(self):
        self.decoder_layers = []

        for i in range(self.num_stacked_rnn):
            cell = layers.LSTM(self.hidden_units, return_sequences=True, name=f"decoder_LSTM_{i}")
            self.decoder_layers.append(cell)

        self.context_repeater = layers.RepeatVector(self.out_steps)
        self.out_fc = layers.Dense(self.output_size)
        self.out = layers.TimeDistributed(self.out_fc)

    def call(self, inputs, training=None, mask=None):
        previous_outs = inputs

        layers_states = []
        for layer_index, encoder_layer in enumerate(self.encoder_layers):
            print(f"Encoder layer {layer_index}")
            print(previous_outs)

            previous_outs, *state = encoder_layer(previous_outs)
            layers_states.append(state)

        # context vector
        previous_outs = self.context_repeater(previous_outs)
        for decoder_layer, encoder_state in zip(self.decoder_layers, layers_states):
            previous_outs = decoder_layer(previous_outs, initial_state=encoder_state)

        forecast = self.out(previous_outs)
        return forecast

class Seq2SeqLSTM2(tf.keras.Model):
    """
    Now explicitly building the graph, the endog variable (the one being forecasted) must be placed first in inputs,
    so for an input vector (batch, N_timesteps, feature_vector) the endog variables index must be [:, :, 0]
    """
    def __init__(self, input_size, output_size, out_steps, hidden_units, num_stacked_rnn):
        """

        :param input_size: (Number of input features per timestep)
        :param output_size: (Number of output features per forecasting timestep)
        :param out_steps: Number of timesteps to forecast in the future
        :param hidden_units:
        :param num_stacked_rnn:
        """
        super(Seq2SeqLSTM2, self).__init__()

        self.out_steps = out_steps
        self.num_stacked_rnn = num_stacked_rnn
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units

        encoder_inputs, encoder_outputs, encoder_states = self._construct_encoder()
        decoder_inputs, decoder_outputs = self._construct_decoder(encoder_inputs, encoder_states)

        self.model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    def _construct_encoder(self):
        encoder_inputs = Input(shape=(None, self.input_size), name="encoder_input")

        # For now only a single LSTM layer
        encoder = layers.LSTM(self.hidden_units, return_state=True, name="encoder_LSTM")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        encoder_states = [state_h, state_c]

        return encoder_inputs, encoder_outputs, encoder_states

    def _construct_decoder(self, encoder_inputs, encoder_states):
        # these 2 lines below should actually be used for inference
        # decoder input is the last timesteps endog feature
        decoder_inputs = tf.gather_nd(params=encoder_inputs, indices=[None, -1, 0])
        # reshape into (batch_dim, 1 or len(timestep output size if needed for multivariate forecasting))
        decoder_inputs = tf.reshape(decoder_inputs, [None, 1])
        # inference lines end


        # training

        decoder = layers.LSTM(self.hidden_units, return_sequences=True, return_state=True, name="decoder_LSTM")
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

        decoder_out_fc = layers.Dense(self.output_size, name="decoder_FC")
        decoder_outputs = decoder_out_fc(decoder_outputs)

        return decoder_inputs, decoder_outputs

    def call(self, inputs, training=None, mask=None):

