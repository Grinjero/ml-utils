import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate_forecast(forecasts, true_values, date_offset, interval=300):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    date_start = forecasts[0].index[0] - date_offset
    date_end = forecasts[0].index[-1] + date_offset

    forecast_line, = ax.plot(forecasts[0].index, forecasts[0].values, label="Forecast", c='orange')
    true_values_lines, = ax.plot(true_values.loc[date_start:date_end].index, true_values.loc[date_start:date_end].values, label= "Original", c='b', alpha=0.7)

    ax.legend()

    def animate(timestep_index):
        forecast_line.set_data(forecasts[timestep_index].index, forecasts[timestep_index].values)

        date_start = forecasts[timestep_index].index[0] - date_offset
        date_end = forecasts[timestep_index].index[-1] + date_offset

        true_values_lines.set_data(true_values.loc[date_start:date_end].index, true_values.loc[date_start:date_end].values)
        ax.set_xlim((date_start, date_end))

    ani = animation.FuncAnimation(fig, animate, interval=interval, frames=len(forecasts), blit=False)
    return ani