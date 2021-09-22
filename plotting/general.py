import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_twin_y_axes(series_1, series_2, color1=None, color2=None, label1=None, label2=None, ax=None):
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

    line1 = _plot_corresponding_to_type(series_1, color=color1, label=label1, ax=ax)
    ax.set_ylabel(label1, color=color1)

    ax2 = ax.twinx()
    line2 = _plot_corresponding_to_type(series_2, ax=ax2, color=color2, label=label2)
    ax2.set_ylabel(label2, color=color2)

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


def _plot_corresponding_to_type(series, ax=None, color=None, label=None):
    if ax is None:
        plotter = plt
    else:
        plotter = ax

    if isinstance(series, pd.Series):
        line = plotter.plot(series.index, series.values, color=color, label=label)

    elif isinstance(series, tuple):
        line = plotter.plot(series[0], series[1], color=color, label=label)

    else:
        line = plotter.plot(series, color=color, label=label)

    return line
