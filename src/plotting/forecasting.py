import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np


def animate_forecast(forecasts, true_values, date_offset, y_lim_mod=0.2, interval=300, ylims=None):
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