import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pandas.tseries.frequencies import to_offset
from mlutils.timeseries import extract_true_values_for_forecast, extract_true_values_for_forecasts
from mlutils.timeseries._operations import align_series_list
import pandas as pd


def animate_forecast(forecasts, true_values, date_offset, y_lim_mod=0.2, interval=300, ylims=None):
    """
    Produces an animation of given forecasts atop the original values.

    :param forecasts:
    :param true_values:
    :param date_offset: date offset left and right to the currently shown forecast
    :param y_lim_mod: percentage of the currently shown `y` amplitude to be added to each side of `ylim`
    :param interval: interval between frames in miliseconds
    :param ylims: Sets `ylim` during the whole animation. If set, ignores `y_lim_mod` and `ylim` is not auto adjusted
        during the animation.
    :return: the animation object that needs to be stored for the duration of the animation (or for however long you
        need it to run)
    """
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    date_start = forecasts[0].index[0] - date_offset
    date_end = forecasts[0].index[-1] + date_offset

    forecast_line, = ax.plot(forecasts[0].index, forecasts[0].values, label="Forecast", c='orange')
    true_values_lines, = ax.plot(true_values.loc[date_start:date_end].index, true_values.loc[date_start:date_end].values, label= "Original", c='b', alpha=0.7)

    ax.legend()

    def animate(timestep_index):
        current_forecast = forecasts[timestep_index]

        date_start = current_forecast.index[0] - date_offset
        date_end = current_forecast.index[-1] + date_offset

        current_true = true_values.loc[date_start:date_end]

        forecast_line.set_data(current_forecast.index, current_forecast.values)

        true_values_lines.set_data(current_true.index, current_true.values)
        ax.set_xlim((date_start, date_end))

        if ylims:
            ax.set_ylim((ylims[0], ylims[1]))
        elif y_lim_mod:
            ymax = max(np.max(current_true.values), np.max(current_forecast.values))
            ymin = min(np.min(current_true.values), np.min(current_forecast.values))
            amp = ymax - ymin

            ymax += amp * y_lim_mod
            ymin -= amp * y_lim_mod
            ax.set_ylim((ymin, ymax))

    ani = animation.FuncAnimation(fig, animate, interval=interval, frames=len(forecasts), blit=False)
    return ani


def plot_forecasts_at_horizon_step(forecasts, true_values, forecast_step):
    """
    :param forecasts:
    :param true_values:
    :type: true_values: pd.Series
    :param forecast_step:
    :return:
    """
    forecasts_step_indices = []
    if isinstance(forecasts, np.ndarray) is False:
        for forecast in forecasts:
            forecasts_step_indices.append(forecast.index[forecast_step])
        forecasts = np.asarray(forecasts)

    horizon_size = forecasts.shape[1]
    assert -horizon_size < forecast_step < horizon_size, "Forecast step must be able to index individual forecasts"

    forecasts_at_step = forecasts[:, forecast_step]

    plt.plot(true_values, label=f"True values", alpha=0.7)
    plt.plot(forecasts_step_indices, forecasts_at_step, label=f"Forecast at step {forecast_step}", alpha=0.7)


def plot_horizon_step_metric(horizon_step_metrics, metric_name):
    x_values = np.arange(1, len(horizon_step_metrics) + 1)

    plt.grid(True)
    plt.plot(x_values, horizon_step_metrics, label=metric_name)
    plt.xlim(1, len(horizon_step_metrics) + 1)


def plot_forecasts_scatterplot(forecasts, true_values, show_legend=False, ax=None):
    """
    Plots the scatterplot
    :param forecasts:
    :param true_values:
    """
    true_value_indices, true_values = extract_true_values_for_forecasts(true_values, forecasts)
    if isinstance(forecasts, np.ndarray) is False:
        forecasts = np.asarray(forecasts)

    horizon_size = forecasts.shape[1]

    if ax is None:
        fig, ax = plt.subplots()

    for step_ind in range(horizon_size):
        true_values_ahead = true_values[:, step_ind]
        step_ahead_forecasts = forecasts[:, step_ind]
        ax.scatter(x=true_values_ahead, y=step_ahead_forecasts, label=f"{step_ind + 1} forecast")

    plt.ylabel("Forecasts")
    plt.xlabel("True")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    if show_legend is True:
        ax.legend()

    return ax


def plot_forecasts_scatterplot_at_horizon_step(forecasts, true_values, forecast_step):
    """

    :param forecasts:
    :type forecasts: array_like of shape (n_forecasts, horizon_size)
    :param true_values:
    :param forecast_step:
    :return:
    """

    _, true_values = extract_true_values_for_forecasts(true_values, forecasts)
    #
    # if isinstance(true_values, np.ndarray) is False:
    #     true_values = np.asarray(true_values)

    if isinstance(forecasts, np.ndarray) is False:
        forecasts = np.asarray(forecasts)

    horizon_size = forecasts.shape[1]

    true_values_ahead = true_values[:, forecast_step]
    step_ahead_forecasts = forecasts[:, forecast_step]

    fig, ax = plt.subplots()
    # ax.scatter(x=true_values_ahead, y=step_ahead_forecasts)
    sns.regplot(x=true_values_ahead, y=step_ahead_forecasts, ax=ax)
    plt.ylabel(f"{forecast_step} step ahead forecast")
    plt.xlabel("True")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)


