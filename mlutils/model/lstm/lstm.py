import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten
from tensorflow.python.keras.layers import TimeDistributed, RepeatVector

# def define_sliding_window_1_layer_cnn(filters=10, kernel_size=5):
#     output_size = future_target
#
#     inputs = Input(shape=window_generator.input_shape)
#     cnn_1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, activation="relu")(inputs)
#     cnn_2 = Conv1D(padding="causal", filters=filters, kernel_size=kernel_size, strides=2, activation="relu")(cnn_1)
#     flattened = Flatten()(cnn_2)
#     output = Dense(output_size)(flattened)
#     return keras.Model(inputs, output), output_size,  "Sliding_window_1_layer_CNN"
#
# def define_1_layer_lstm_seq_2_seq(num_units=10):
#
#     inputs = Input(shape=window_generator.input_shape)
#     encoder_lstm = LSTM(num_units, activation='tanh', recurrent_activation='sigmoid',
#                         return_sequences=False, name="encoder_lstm")(inputs)
#
#     encoder_lstm = tf.reshape(encoder_lstm, (-1, 1, num_units))
#     # encoding_repeat = layers.RepeatVector(future_target)(encoder_lstm)
#     pad_constant = tf.constant([[0, 0], [0, future_target-1], [0, 0]], dtype=tf.int32)
#     padded_encoder = tf.pad(encoder_lstm, paddings=pad_constant, mode="CONSTANT")
#
#     decoder_lstm = LSTM(num_units, activation='tanh', recurrent_activation='sigmoid',
#                         return_sequences=True, name="decoder_lstm")(padded_encoder)
#     sequence_prediction = TimeDistributed(Dense(1, activation=None, name="out_dense"))(decoder_lstm)
#
#     return keras.Model(inputs, sequence_prediction), future_target, "LSTM_seq_2_seq"
#
#
# def define_2_layer_lstm(num_units=10):
#     model = tf.keras.models.Sequential()
#     model.add(Input(shape=window_generator.input_shape))
#
#     model.add(layers.LSTM(num_units, activation='tanh', recurrent_activation='sigmoid',
#                           return_sequences=True))
#     model.add(layers.Dropout(0.1))
#     model.add(layers.LSTM(num_units, activation='tanh', recurrent_activation='sigmoid'))
#     model.add(layers.Dense(future_target))
#
#     return model, future_target, "2_LSTM_one_shot"
#
# def define_1_layer_lstm(num_units=10):
#     inputs = Input(shape=window_generator.input_shape)
#     lstm = LSTM(num_units, activation='tanh', recurrent_activation='sigmoid')(inputs)
#     dropout = layers.Dropout(0.1)(lstm)
#     dense_out = Dense(future_target)(dropout)
#
#     model = tf.keras.Model(inputs, dense_out)
#     return model, future_target, "1_LSTM_one_shot"


def hyperparameter_search_one_shot_LSTM(hp, input_shape, config, num_layers, num_dense_layers, min_units, max_units, steps):
    inputs = Input(shape=input_shape)
    previous_out = inputs
    for layer_i in range(hp.Int("num_layers", min_value=1, max_value=num_layers - 1)):
        lstm = LSTM(hp.Int("lstm_units_{}".format(layer_i), min_value=min_units, max_value=max_units, step=steps),
                    activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name="lstm_{}".format(layer_i))(previous_out)
        dropout = Dropout(0.1, name="dropout_encoder_{}".format(layer_i))(lstm)

        previous_out = dropout

    lstm = LSTM(hp.Int("lstm_units_{}".format(num_layers - 1), min_value=min_units, max_value=max_units, step=steps),
                activation='tanh', recurrent_activation='sigmoid', return_sequences=False,
                name="lstm_{}".format(num_layers - 1))(previous_out)
    dropout = Dropout(0.1, name="dropout_encoder_{}".format(num_layers - 1))(lstm)
    previous_out = dropout

    for dense_layer_i in range(hp.Int("num_dense_layers", min_value=0, max_value=num_dense_layers)):
        dense_decoder = (Dense(hp.Int("dense_units_{}".format(dense_layer_i), min_value=min_units, max_value=max_units, step=steps),
                               activation="relu", name="dense_decoder_{}".format(dense_layer_i)))(previous_out)
        dropout_decoder = Dropout(0.1, name="dropout_decoder_{}".format(dense_layer_i))(dense_decoder)

        previous_out = dropout_decoder
    dense_out = Dense(config["future_target"], name="dense")(previous_out)

    model = tf.keras.Model(inputs, dense_out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["initial_lr"]),
                  loss=config["loss_method"],
                  metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])
    return model


def one_shot_LSTM(input_shape, future_target, num_units=[15, 20], dense_units=[]):
    inputs = Input(shape=input_shape)

    previous_out = inputs
    for counter, units in enumerate(num_units):
        ret_seq = counter != (len(num_units) - 1)
        lstm = LSTM(units, activation='tanh', recurrent_activation='sigmoid', return_sequences=ret_seq, name="lstm_{}".format(counter))(previous_out)
        dropout = Dropout(0.1, name="dropout_encoder_{}".format(counter))(lstm)

        previous_out = dropout
    for counter, dense_unit in enumerate(dense_units):
        dense_decoder = (Dense(dense_unit, activation="relu", name="dense_decoder_{}".format(counter)))(previous_out)
        dropout_decoder = Dropout(0.1, name="dropout_decoder_{}".format(counter))(dense_decoder)

        previous_out = dropout_decoder
    dense_out = Dense(future_target, name="dense")(previous_out)

    model = tf.keras.Model(inputs, dense_out)
    return model, "{}_LSTM_one_shot".format(len(num_units))


def autoregressive_LSTM(input_shape, future_target, encoder_units=[15, 40], decoder_units=[20, 30], dense_units=[]):
    inputs = Input(shape=input_shape)

    previous_out = inputs
    for counter, encoder_unit in enumerate(encoder_units):
        ret_seq = counter != (len(encoder_units) - 1)
        encoder_lstm = LSTM(encoder_unit, activation='tanh', recurrent_activation='sigmoid',
                            return_sequences=ret_seq, name="encoder_lstm_{}".format(counter))(previous_out)
        encoder_dropout = Dropout(0.1, name="encoder_dropout_{}".format(counter))(encoder_lstm)

        previous_out = encoder_dropout

    # encoded = tf.reshape(previous_out, (-1, 1, encoder_units[-1]))
    # print(tf.shape(encoded))
    encoding_repeat = RepeatVector(future_target)(previous_out)
    # pad_constant = tf.constant([[0, 0], [0, future_target - 1], [0, 0]], dtype=tf.int32)
    # padded_encoder = tf.pad(encoded, paddings=pad_constant, mode="CONSTANT")

    previous_out = encoding_repeat
    for counter, decoder_unit in enumerate(decoder_units):
        # ret_seq = counter != (len(decoder_units) - 1)
        decoder_lstm = LSTM(decoder_unit, activation='tanh', recurrent_activation='sigmoid',
                            return_sequences=True, name="decoder_lstm_{}".format(counter))(previous_out)
        decoder_dropout = Dropout(0.1, name="decoder_dropout_{}".format(counter))(decoder_lstm)

        previous_out = decoder_dropout

    for counter, dense_unit in enumerate(dense_units):
        dense_decoder = TimeDistributed(
            Dense(dense_unit, activation="relu", name="dense_decoder_{}".format(counter)))(previous_out)

        previous_out = dense_decoder

    sequence_prediction = TimeDistributed(Dense(1, activation=None, name="out_dense"))(previous_out)
    model = tf.keras.Model(inputs, sequence_prediction)
    return model, "{}_{}_LSTM_seq_2_seq".format(len(encoder_units), len(decoder_units))

