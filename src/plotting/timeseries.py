import matplotlib.pyplot as plt

from statsmodels.tsa import seasonal


def plot_series_segments(series, index_pairs):
    """
    Plots the segments of the given series defined by `index_pairs`

    :param series: pd.Series whose segments will be plotted.
    :param index_pairs: list of tuples (`segment_index_start`, `segment_index_end`) defining each segment
    """
    for index_start, index_end in index_pairs:
        segment = series.loc[index_start:index_end]
        segment.plot()


def plot_seasonal_decompose(decompose: seasonal.DecomposeResult, figsize=(20, 12), period_name=None):
    """
    Plotting of a seasonal decomposition on 4 axes, one for each component. Components are plotted in the order: observed,
    seasonal, trend and residuals. plt.show() still needs to be called after this function.

    :param decompose: result of a statsmodels decomposition (STL, seasonal_decompose, etc.)
    :param figsize: figsize of the whole plot
    :param period_name: Used to adjust titles, examples are "Weekly", "Daily", etc.
    :return: tuple of 4 axes
    """
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=figsize)

    ax0.plot(decompose.observed)
    ax0.set_ylabel("Observed")

    ax1.plot(decompose.seasonal)
    ax1.set_ylabel(f"{period_name} seasonal")

    ax2.plot(decompose.trend)
    ax2.set_ylabel(f"{period_name} trend")

    ax3.plot(decompose.resid)
    ax3.set_ylabel(f"{period_name} residuals")

    return ax0, ax1, ax2, ax3
