def calculate_metric_dict(predictions, true_values, metrics):
    """
    Calculates each given metric for given predictions and their true values.

    Use case: in case of running multiple different experiments at the top of the file you can easily define your set of
     metrics and easily store them. If at a later point you decide to change your metrics you need only change a single
     dict
    :param predictions:
    :param true_values:
    :param metrics: dict with key-value pairs {metric name}-{metric func} that you want calculated.
    Each metric func should receive parameters (predictions, true_values)
    """

    metric_dict = {}
    for metric_name, metric_func in metrics:
        metric_value = metric_func(predictions, true_values)

        if hasattr(metric_value, '__len__'):
            # In case the metric returns multiple values
            for i, value in enumerate(metric_value):
                metric_dict[f"{metric_name}_{i}"] = value
        else:
            metric_dict[metric_name] = metric_value

    return metric_dict
