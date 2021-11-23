import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import tensorflow as tf


class WindowGenerator:
    def __init__(self, sequence_length, target_sequence_length, shift, sequence_stride, train_df, val_df, test_df,
                 feature_columns,
                 label_columns, shuffle_size=32, batch_size=8):
        """
        :param sequence_length: time length of the input timesequence
        :param target_sequence_length: time length of the output target timesequence
        :param shift: time shift between the first input element and the first target element
        :param train_df: train data DataFrame
        :param val_df: validation data DataFrame
        :param test_df: test data DataFrame
        :param feature_columns: columns to be used as input features, column names
        :param label_columns: columns to be used as target features, column names
        """

        assert sequence_stride >= 1

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

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
        self.labels_slice = slice(self.label_start, self.total_window_size, sequence_stride)
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

    def make_dataset(self, data):
        features = np.array(data[self.feature_column_names], dtype=np.float32)
        targets = np.array(data[self.label_column_names], dtype=np.float32)

        assert len(features) == len(targets)

        start_positions = np.arange(start=0, stop=len(features) - self.total_window_size - 1, step=self.sequence_stride)
        slices = [np.arange(start=start_position, stop=start_position + self.total_window_size)
                  for start_position in start_positions]
        input_indices = [sample_slice[self.input_slice] for sample_slice in slices]
        label_indices = [sample_slice[self.labels_slice] for sample_slice in slices]

        inputs = features[input_indices, :]
        labels = targets[label_indices, :]

        return tf.data.Dataset.from_tensor_slices((inputs, labels))

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
        return chosen_set[selected_columns].iloc[
               start_index * self.sequence_stride:end_index * self.sequence_stride].to_numpy()

    def get_values_as_timeseries(self, selected_column, selected_set, return_indices=False):
        """
        :param selected_column: str or list of str
        :param selected_set:
        :param return_indices: also corresponding indices for x and y
        :return:
        """
        chosen_set = self._choose_set(selected_set)

        index = chosen_set.index
        values = chosen_set[selected_column].to_numpy()
        history_timeseries = []
        target_timeseries = []

        if return_indices:
            history_loc_list = []
            target_loc_list = []

        n = len(values)
        for i in range(0, n - self.total_window_size):
            window_indices = self.get_window_indices(i, True)
            history_indices, target_indices = window_indices[self.input_indices], window_indices[self.label_indices]
            histories, targets = values[history_indices], values[target_indices]

            if return_indices:
                history_index, target_index = index[history_indices], index[target_indices]
                history_loc_list.append(history_index)
                target_loc_list.append(target_index)

            history_timeseries.append(histories)
            target_timeseries.append(targets)

        history_timeseries = np.stack(history_timeseries)
        target_timeseries = np.stack(target_timeseries)

        if return_indices:
            return history_timeseries, target_timeseries, history_loc_list, target_loc_list
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

    def apply_model(self, model, selected_set="test", return_indices=False):
        values = self.get_values_as_timeseries(self.feature_column_names, selected_set, return_indices)
        x, y_true = values[0], values[1]

        predictions = model(x).numpy()

        if return_indices:
            indexed_predictions = []

            predictions_indices = values[-1]
            for single_forecast, single_forecast_index in zip(predictions, predictions_indices):
                if len(self.label_column_names) == 1:
                    indexed_prediction = pd.Series(index=single_forecast_index, data=single_forecast,
                                                   name=self.label_column_names[0])
                else:
                    indexed_prediction = pd.DataFrame(index=single_forecast_index, data=single_forecast,
                                                      columns=self.label_column_names)

                indexed_predictions.append(indexed_prediction)

            return indexed_predictions

        else:
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
                add_data = {column: chosen_set.iloc[window_indices][column].to_numpy() for column in chosen_set.columns}
                samples.append((inputs, labels, add_data))
            else:
                samples.append((inputs, labels))

        return samples

    # def apply_inverse_transformation(self, outputs, data, feature_name, indices=None):
    #     transformation_func, orig_feature_name = self.inverse_transformations[feature_name]
    #
    #     outputs = outputs.reshape(-1, 1)
    #
    #     if isinstance(data, dict):
    #         orig = data[orig_feature_name]
    #     else:
    #         if orig_feature_name in self.feature_column_indices:
    #             orig_feature_ind = self.feature_column_indices[orig_feature_name]
    #             orig = data[:, orig_feature_ind].reshape(-1, 1)
    #         else:
    #             orig = data
    #     if indices is not None:
    #         orig = orig[indices]
    #     return transformation_func(outputs, orig)


if __name__ == "__main__":
    df = pd.read_csv("D:/Faks/T-LOGIC/SalesPrediction/data/abc_xyz_machine_sales.csv")
    df["vend_date"] = pd.to_datetime(df["vend_date"])
    machine_products_combos = [
        (1623, 35351),
        (1432, 3353),
        (838, 3979),
        (1269, 3995),
        (1610, 3961),
        (42413, 4446),
        (38130, 77122),
        (38364, 77118)
    ]


    def asfreq_0_holes(group):
        asf = group.set_index("vend_date", drop=True).sort_index().asfreq("D")

        columns_to_0 = ["daily_quantity", "daily_value"]
        columns_others = list(set(asf.columns) - set(columns_to_0))

        asf.loc[:, columns_others] = asf.loc[:, ["machine_id", "product_id"]].ffill()
        asf.loc[:, columns_to_0] = asf.loc[:, ["daily_quantity", "daily_value"]].fillna(0)
        return asf.reset_index(drop=False)


    print("Started dropping combos")
    df = df[df[["product_id", "machine_id"]].apply(
        lambda row: (row.machine_id, row.product_id) in machine_products_combos, axis=1)]
    print("Dropped all other combos")
    df = df.groupby(["machine_id", "product_id"]).apply(
        asfreq_0_holes
    ).reset_index(drop=True)
    print("Filled sale holes")
    generator_dict = {}

    train_data_percentage = 0.4
    val_data_percentage = 0.3
    history = 14
    future = 7
    sequence_stride = 1
    label_shape = None
    input_shape = None

    used_feature_names = ["daily_quantity"]
    target_feature_name = ["daily_quantity"]

    for (machine_id, product_id), group in df.groupby(["machine_id", "product_id"]):
        train_group = group.set_index("vend_date").asfreq("D")
        n = len(group)
        val_start_index = int(n * train_data_percentage)
        test_start_index = int(n * (train_data_percentage + val_data_percentage))

        generator_dict[(machine_id, product_id)] = WindowGenerator(history, future, shift=history,
                                                                   train_df=train_group.iloc[0:val_start_index],
                                                                   val_df=train_group.iloc[
                                                                          val_start_index:test_start_index],
                                                                   test_df=train_group.iloc[test_start_index:],
                                                                   sequence_stride=sequence_stride,
                                                                   feature_columns=used_feature_names,
                                                                   label_columns=target_feature_name)
        label_shape = generator_dict[(machine_id, product_id)].label_shape
        input_shape = generator_dict[(machine_id, product_id)].input_shape

    for machine_id, product_id in generator_dict.keys():
        window_generator = generator_dict[(machine_id, product_id)]

        for input, label in window_generator.train:
            assert label.shape[:-2] == label_shape
            assert input.shape[:-2] == input_shape
        for input, label in window_generator.val:
            assert label.shape[:-2] == label_shape
            assert input.shape[:-2] == input_shape
        for input, label in window_generator.test:
            assert label.shape[:-2] == label_shape
            assert input.shape[:-2] == input_shape

    print("Done all labels are of same shape")
