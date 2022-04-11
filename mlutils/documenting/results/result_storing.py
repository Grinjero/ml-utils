import os

import pandas as pd
import numpy as np
import datetime

from functools import reduce
from operator import or_ as union

from utils.constants import MODELS_FOLDER, FORECASTS_FOLDER


def get_forecast_storage_directory_path(model):
    directory_path = os.path.join(FORECASTS_FOLDER, model)
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def get_result_file_name(input_path):
    _, file_name = os.path.split(input_path)
    return os.path.splitext(file_name)[0]


def get_result_storage_file_path(model, result_name, append_timestamp=False):
    dir = get_forecast_storage_directory_path(model)

    file_name = result_name
    if append_timestamp:
        file_name += "_"
        file_name += datetime.datetime.now().strftime("%d_%m_%H-%M")

    file_name += ".pkl"
    return os.path.join(dir, file_name)


def create_filename(base_name=None, append_timestamp=True):
    """
    Withot extension
    """
    assert (base_name is not None) and (append_timestamp is not False), ("Must give atleast one of the input arguments")

    file_name = ""
    if base_name is not None:
        file_name += base_name
    if append_timestamp is True:
        file_name += "_"
        file_name += datetime.datetime.now().strftime("%d_%m_%H-%M")
    return file_name


class MultiInstanceResultStorer:
    """
    CSV files saved using this class contain lines of metadata on the beginning of each file. Metadata lines start with
    # so if someone is interested in loading these files with pd.read_csv be sure to use the `comment='#'` argument.
    """
    def __init__(self):
        """
        :param true_values_dataset: indexed by (Instance instance keys, time), time must be last in line
        """
        self.instance_dict = {}

    def store_instance_forecasts(self, forecasts, instance_keys):
        if instance_keys not in self.instance_dict:
            self.instance_dict[instance_keys] = InstanceResultStorer()

        self.instance_dict[instance_keys].store_instance_forecasts(forecasts)

    def get_dict_format(self):
        return {instance_keys: instance_results.get_list_format()
                for instance_keys, instance_results in self.instance_dict.items()}

    def get_df_format(self):
        instance_dfs = []
        instance_keys = []

        for instance_key, instance_results in self.instance_dict.items():
            instance_keys.append(instance_key)
            instance_dfs.append(instance_results.get_df_format())

        return pd.concat(instance_dfs, keys=instance_keys)

    def to_csv(self, path):
        self.get_df_format().to_csv(path)

    def to_pkl(self, path):
        self.get_df_format().to_pickle(path)


class InstanceResultStorer:
    def __init__(self):
        """
        :param true_values_dataset: indexed by (Instance instance keys, time), time must be last in line
        """
        self.forecast_list = []

    def store_instance_forecasts(self, forecasts):
        self.forecast_list.extend(forecasts)

    def get_list_format(self):
        return self.forecast_list

    def _create_index_union(self):
        list_of_forecast_indices = [forecast.index for forecast in self.forecast_list]
        return reduce(union, list_of_forecast_indices)

    def get_df_format(self):
        df_index = self._create_index_union()
        forecasts_df = pd.DataFrame(index=df_index)

        forecasts_indices = []
        for forecast in self.forecast_list:
            forecasts_indices.append(forecast.index)
        # rows: individual forecast horizons, cols: forecast offsets
        forecasts_indices = np.asarray(forecasts_indices)
        forecasts = np.asarray(self.forecast_list)

        horizon_size = forecasts_indices.shape[1]
        for horizon_step in range(horizon_size):
            indices = forecasts_indices[:, horizon_step]
            forecasts_df.loc[indices, horizon_step] = forecasts[:, horizon_step]

        return forecasts_df

    def to_csv(self, path):
        self.get_df_format().to_csv(path)

    def to_pkl(self, path):
        self.get_df_format().to_pickle(path)




