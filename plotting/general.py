import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def _construct_plotting_kwargs(**kwargs):
    plot_kwargs = dict()
    if 'color' in kwargs and kwargs["color"] is not None:
        plot_kwargs["color"] = kwargs["color"]
    if 'label' in kwargs and kwargs["label"] is not None:
        plot_kwargs["label"] = kwargs["label"]

    plot_kwargs.update(kwargs)
    return plot_kwargs


def plot_twin_y_axes(series_1, series_2, label1, label2, color1, color2, ax=None, **kwargs):
    """
    Useful for plotting data that needs to be observed on the same graph, but that "lives" in different scales
    etc. mean and std of a variable
    :param series_1: a tuple (x axis, y axis), pd.Series or simple array plotted on the left
    :param series_2: a tuple (x axis, y axis), pd.Series or simple array plotted on the right
    :param color1:
    :param color2:
    :param ax:
    :return:
    """

    if ax is None:
        fig, ax = plt.subplots()
    #
    # plot_kwargs1 = _construct_plotting_kwargs(label=label1, color=color1, **kwargs).
    # plot_kwargs2 = _construct_plotting_kwargs(label=label2, color=color2, **kwargs)

    line1 = _plot_corresponding_to_type(series_1, color=color1, label=label1, ax=ax, **kwargs)
    ax.set_ylabel(ylabel=label1, color=color1)

    ax2 = ax.twinx()
    line2 = _plot_corresponding_to_type(series_2, ax=ax2, color=color2, label=label2, **kwargs)
    ax2.set_ylabel(ylabel=label2, color=color2)

    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs)
    plt.tight_layout()
    return ax


def plot_correlation_matrix(df, figsize=None):
    corr = df.corr()
    # mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    if figsize:
        f, ax = plt.subplots(figsize=(14, 14))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, square=True, center=0,
    #             annot=True)
    sns.heatmap(corr, cmap=cmap, square=True, center=0,
                annot=True)
    plt.yticks(np.arange(len(corr)) + 0.5, va="center")

    if figsize:
        return ax


def _plot_corresponding_to_type(series, ax=None, **kwargs):
    if ax is None:
        plotter = plt
    else:
        plotter = ax

    # kwargs = dict()
    # if color:
    #     kwargs["color"] = color
    # if label:
    #     kwargs["label"] = label

    if isinstance(series, pd.Series):
        line = plotter.plot(series.index, series.values, **kwargs)

    elif isinstance(series, tuple):
        line = plotter.plot(series[0], series[1], **kwargs)

    else:
        line = plotter.plot(series, **kwargs)

    return line
