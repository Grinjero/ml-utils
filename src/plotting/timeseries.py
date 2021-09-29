

def plot_series_segments(series, index_pairs):
    """
    Plots the segments of the given series defined by `index_pairs`

    :param series: pd.Series whose segments will be plotted.
    :param index_pairs: list of tuples (`segment_index_start`, `segment_index_end`) defining each segment
    """
    for index_start, index_end in index_pairs:
        segment = series.loc[index_start:index_end]
        segment.plot()
