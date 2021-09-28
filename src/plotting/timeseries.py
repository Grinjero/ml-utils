

def plot_series_segments(series, index_pairs):
    for index_start, index_end in index_pairs:
        segment = series.loc[index_start:index_end]
        segment.plot()
