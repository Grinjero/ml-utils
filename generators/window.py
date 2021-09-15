import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


class WindowGenerator:
    def __init__(self, sequence_length, target_sequence_length, shift, sequence_stride, train_df, val_df, test_df,
                 feature_columns,
                 label_columns, inverse_transformations, shuffle_size=32, batch_size=8):
        """
        :param sequence_length: time length of the input timesequence
        :param target_sequence_length: time length of the output target timesequence
        :param shift: time shift between the first input element and the first target element
        :param train_df: train data DataFrame
        :param val_df: validation data DataFrame
        :param test_df: test data DataFrame
        :param feature_columns: columns to be used as input features, column names
        :param label_columns: columns to be used as target features, column names
        :param inverse_transformations: dict that transformations of certain features back to their original state
        (mostly used to transform predictions back their real world representation).
        Each value is a tuple(transformation_func, original feature name)
        Function arguments are (data to be transformed, input features)
        """

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.inverse_transformations = inverse_transformations

        self.sequence_length = sequence_length
        self.target_sequence_length = target_sequence_length
        self.shift = shift
        self.sequence_stride = sequence_stride
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size

        self.column_name_index = {name: i for i, name in enumerate(train_df.columns)}
        if isinstance(feature_columns[0], str):
            self.feature_column_indices = {feature_column_name: counter for counter, feature_column_name in
                                           enumerate(feature_columns)}
            self.feature_column_names = feature_columns
        else:
            raise TypeError("Feature columns must contain elements of type int or str, received {}".format(
                type(feature_columns[0])))

        if isinstance(label_columns[0], str):
            self.label_column_indices = {label_column_name: counter for counter, label_column_name in
                                         enumerate(label_columns)}
            self.label_column_names = label_columns
        else:
            raise TypeError(
                "Feature columns must contain elements of type int or str, received {}".format(type(label_columns[0])))

        self.total_window_size = (max(sequence_length, shift + target_sequence_length)) * self.sequence_stride

        self.input_slice = slice(0, sequence_length * self.sequence_stride, sequence_stride)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.target_sequence_length * sequence_stride
        self.labels_slice = slice(self.label_start, -1, sequence_stride)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, x, y):
        inputs = x[:, self.input_slice, len(self.feature_column_indices)]
        labels = y[:, self.labels_slice, len(self.label_column_indices)]

        inputs.set_shape([None, self.sequence_length, None])
        labels.set_shape([None, self.target_sequence_length, None])

        return inputs, labels

    def get_starting_position_input_indices(self, starting_position):
        rg = range(starting_position, starting_position + self.total_window_size, self.sequence_stride)
        return rg[self.input_slice], rg[self.labels_slice]

    def get_window_indices(self, starting_position, full_window=False):
        if full_window:
            return np.arange(starting_position, starting_position + self.total_window_size)
        else:
            return np.arange(starting_position, starting_position + self.total_window_size, self.sequence_stride)

    # def make_dataset(self, data):
    #     features = np.array(data[self.feature_column_names], dtype=np.float32)
    #     targets = np.array(data[self.label_column_names], dtype=np.float32)
    #
    #     assert len(features) == len(targets)
    #
    #     start_positions = np.arange(start=0, stop=len(features) - self.total_window_size, step=self.sequence_stride)
    #     slices = [np.arange(start_position, start_position + self.total_window_size)
    #               for start_position in start_positions]
    #     input_indices = [sample_slice[self.input_slice] for sample_slice in slices]
    #     label_indices = [sample_slice[self.labels_slice] for sample_slice in slices]
    #
    #     inputs = features[input_indices, :]
    #     labels = targets[label_indices, :]
    #
    #     return tf.data.Dataset.from_tensor_slices((inputs, labels))

    @property
    def input_shape(self):
        return self.sequence_length, len(self.feature_column_names)

    @property
    def label_shape(self):
        return self.target_sequence_length, len(self.label_column_indices)

    @property
    def num_features(self):
        return len(self.feature_column_names)

    @property
    def num_labels(self):
        return len(self.label_column_names)

    @property
    def train(self):
        return self.make_dataset(self.train_df).cache().shuffle(self.shuffle_size).batch(self.batch_size)

    @property
    def val(self):
        return self.make_dataset(self.val_df).cache().batch(self.batch_size)

    @property
    def test(self):
        return self.make_dataset(self.test_df).cache()

    def get_values(self, start_index, end_index, selected_columns, selected_set="train"):
        chosen_set = self._choose_set(selected_set)
        return chosen_set[selected_columns].iloc[start_index * self.sequence_stride:end_index * self.sequence_stride].to_numpy()

    def get_values_as_timeseries(self, selected_column, selected_set):
        """
        :param selected_column: str or list of str
        :param selected_set:
        :return:
        """
        chosen_set = self._choose_set(selected_set)

        values = chosen_set[selected_column].to_numpy()
        history_timeseries = []
        target_timeseries = []
        n = len(values)
        for i in range(0, n - self.total_window_size):
            window_indices = self.get_window_indices(i, True)
            history_indices, target_indices = window_indices[self.input_indices], window_indices[self.label_indices]
            histories, targets = values[history_indices], values[target_indices]

            history_timeseries.append(histories)
            target_timeseries.append(targets)

        history_timeseries = np.stack(history_timeseries)
        target_timeseries = np.stack(target_timeseries)
        return history_timeseries, target_timeseries

    def _choose_set(self, selected_set):
        assert selected_set in ["train", "val", "test"]
        if selected_set == "train":
            chosen_set = self.train_df
        elif selected_set == "test":
            chosen_set = self.test_df
        else:
            chosen_set = self.val_df

        return chosen_set

    def extract_values_from_windows(self, x, indices):
        extracted_x = x[:, indices, :]
        return np.array(extracted_x)


    def extract_inputs_from_windows(self, x):
        return self.extract_values_from_windows(x, self.input_indices)


    def extract_targets_from_windows(self, x):
        return self.extract_values_from_windows(x, self.label_indices)


    def apply_model(self, model, selected_set="test", transform_labels=False):
        x, y_true = self.get_values_as_timeseries(self.feature_column_names, selected_set)
        # x = self.extract_inputs_from_windows(x)

        predictions = model(x).numpy()
        if transform_labels:

            predictions = [
                self.apply_inverse_transformation(sample_predictions, input_features, self.label_column_names[0]) for
                sample_predictions, input_features in zip(predictions, x)]

        return np.stack(predictions)


    def example(self, n=5, example_set="test", include_other=False, full_window=False):
        """
        :param n:
        :param example_set:
        :param include_other: inclued other values as dict
        :param full_window:
        :return:
        """
        chosen_set = self._choose_set(example_set)

        input_data = chosen_set[self.feature_column_names]
        label_data = chosen_set[self.label_column_names]

        samples = []

        for i in range(n):
            rand_start_position = random.randint(0, len(chosen_set) - self.total_window_size)

            window_indices = self.get_window_indices(rand_start_position, full_window)
            if full_window is False:
                input_indices, labels_indices = window_indices[self.input_indices], window_indices[self.label_indices]
            else:
                input_indices, labels_indices = window_indices, window_indices
            inputs, labels = input_data.iloc[input_indices], label_data.iloc[labels_indices]
            inputs, labels = np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)
            if include_other:
                add_data = {column: chosen_set.iloc[window_indices][column].to_numpy() for column in chosen_set.columns }
                samples.append((inputs, labels, add_data))
            else:
                samples.append((inputs, labels))

        return samples

    def apply_inverse_transformation(self, outputs, data, feature_name, indices=None):
        transformation_func, orig_feature_name = self.inverse_transformations[feature_name]

        outputs = outputs.reshape(-1, 1)

        if isinstance(data, dict):
            orig = data[orig_feature_name]
        else:
            if orig_feature_name in self.feature_column_indices:
                orig_feature_ind = self.feature_column_indices[orig_feature_name]
                orig = data[:, orig_feature_ind].reshape(-1, 1)
            else:
                orig = data
        if indices is not None:
            orig = orig[indices]
        return transformation_func(outputs, orig)


if __name__ == "__main__":
    dataset = read_location_flow_csv("Sisak", "Žitna")
    dataset["date"] = pd.to_datetime(dataset["date"])
    batch_size = 8
    shuffle_dim = 64

    n = len(dataset)
    val_start_index = int(n * 0.6)
    test_start_index = int(n * (0.8))
    window_generator = WindowGenerator(20, 10, 20,
                                       train_df=dataset.iloc[0:val_start_index],
                                       val_df=dataset.iloc[val_start_index:test_start_index],
                                       test_df=dataset.iloc[test_start_index:],
                                       sequence_stride=1,
                                       feature_columns=["jam_factor", "speed_cut"],
                                       label_columns=["jam_factor"])

    train = window_generator.train

    print(train.take(1))
    window_generator.plot(title="Traffic jam")
