import pandas as pd
from mlutils.metrics.forecasting.metric_aggregation import agg_horizon_step_metrics, agg_forecast_origin_metrics, agg_horizon_step_metrics_masked, agg_forecast_origin_metrics_mask


def merge_instance_horizon_step_metrics(instances_step_metrics, keys):
    return pd.concat(instances_step_metrics, axis=0, keys=keys)


def merge_instance_forecast_origin_metrics(instances_origin_metrics, keys):
    return pd.concat(instances_origin_metrics, axis=0, keys=keys)


class MultiInstanceMetricStorer:
    def __init__(self, metric_dict):
        self.metric_calculator = MetricCalculator(metric_dict)

        self.instance_wise_metrics = {}
        self.instance_wise_horizon_step_metrics = {}
        self.instance_wise_origin_metrics = {}

    def calculate_instance_metrics(self, true, forecasts_list, instance_key, **metric_kwargs):
        instance_metrics = self.metric_calculator.calculate_metrics(true, forecasts_list, **metric_kwargs)

        self.instance_wise_metrics[instance_key] = instance_metrics
        return instance_metrics

    def calculate_instance_horizon_step_metrics(self, true, forecasts_list, instance_key, **agg_kwargs):
        instance_horizon_step_metrics = self.metric_calculator.calculate_horizon_step_metrics(true, forecasts_list,
                                                                                              **agg_kwargs)

        self.instance_wise_horizon_step_metrics[instance_key] = instance_horizon_step_metrics
        return instance_horizon_step_metrics

    def calculate_instance_forecast_origin_metrics(self, true, forecasts_list, instance_key, **agg_kwargs):
        instance_origin_metrics = self.metric_calculator.calculate_forecast_origin_metrics(true, forecasts_list,
                                                                                           **agg_kwargs)

        self.instance_wise_origin_metrics[instance_key] = instance_origin_metrics
        return instance_origin_metrics

    def calculate_instance_all(self, true, forecasts_list, instance_key, **agg_kwargs):
        instance_metrics = self.calculate_instance_metrics(true, forecasts_list, instance_key)
        instance_horizon_metrics = self.calculate_instance_horizon_step_metrics(true, forecasts_list, instance_key,
                                                                                **agg_kwargs)
        instance_origin_metrics = self.calculate_instance_forecast_origin_metrics(true, forecasts_list, instance_key,
                                                                                  **agg_kwargs)

        return instance_metrics, instance_horizon_metrics, instance_origin_metrics

    def metrics_to_df(self):
        return pd.DataFrame.from_dict(self.instance_wise_metrics, orient="index")

    def aggregate_origin_metrics(self, agg_func):
        origin_metrics = self.origin_metrics_to_df()


    def horizon_step_metrics_to_df(self):
        instance_keys = list(self.instance_wise_horizon_step_metrics.keys())
        step_metric_dfs = [self.instance_wise_horizon_step_metrics[key] for key in instance_keys]
        return merge_instance_horizon_step_metrics(step_metric_dfs, instance_keys)

    def origin_metrics_to_df(self):
        instance_keys = list(self.instance_wise_origin_metrics.keys())
        origin_metric_dfs = [self.instance_wise_origin_metrics[key] for key in instance_keys]
        return merge_instance_forecast_origin_metrics(origin_metric_dfs, instance_keys)


    def _instance_key_shape(self):
        instance_keys = list(self.instance_wise_origin_metrics.keys())


class MetricCalculator:
    def __init__(self, metric_dict):
        """

        Parameters
        ----------
        metric_dict: dict
            key - metric name
            value - metric_func(true, predicted)
        """

        self.metric_dict = metric_dict

    def calculate_metrics(self, true, predicted, **metric_kwargs):
        result_dict = {}
        # TODO: for now we just do an align of true and predicted, we should discuss exact shapes of the input
        # TODO: this function could be used to calculate metrics averaged across every horizon step and forecast origin
        true, predicted = true.align(predicted, join='inner', axis=0)

        for metric_name, metric_func in self.metric_dict.items():
            metric_res = metric_func(true, predicted)
            result_dict[metric_name] = metric_res

        return result_dict

    def calculate_horizon_step_metrics(self, true, forecasts_list, **agg_kwargs):
        result_dict = {}

        horizon_length = None
        for metric_name, metric_func in self.metric_dict.items():
            stepwise_metric = agg_horizon_step_metrics_masked(forecasts_list, true, metric_func, **agg_kwargs)

            result_dict[metric_name] = stepwise_metric

            horizon_length = len(stepwise_metric)

        horizon_steps_df = pd.DataFrame(index=list(result_dict.keys()), columns=range(1, 1 + horizon_length))
        for metric_name in result_dict.keys():
            for step in range(horizon_length):
                horizon_steps_df.loc[metric_name, step + 1] = result_dict[metric_name][step]

        return horizon_steps_df

    def calculate_forecast_origin_metrics(self, true, forecasts_list, **agg_kwargs):
        result_dict = {}

        for metric_name, metric_func in self.metric_dict.items():
            originwise_metric = agg_forecast_origin_metrics_mask(forecasts_list, true, metric_func, **agg_kwargs)

            result_dict[metric_name] = originwise_metric

        indices = []
        for forecast in forecasts_list:
            indices.append(forecast.index[0])

        origin_df = pd.DataFrame(index=indices, columns=list(result_dict.keys()))
        for metric_name in result_dict.keys():
            for single_origin_index, single_origin_metric in result_dict[metric_name].iteritems():
                origin_df.loc[single_origin_index, metric_name] = single_origin_metric

        return origin_df
