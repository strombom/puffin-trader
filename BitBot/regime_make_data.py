
import pickle
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from enum import IntEnum

from BinanceSim.binance_simulator import BinanceSimulator
from Common.Misc import PositionDirection
from Indicators.supersmoother import super_smoother
from position import Position
from regime_plotter import Plotter


def make_spectrum(lengths, prices, poly_order, volatilities, directions):
    for length_idx, length in enumerate(lengths):
        vols = []
        for idx in range(lengths[-1], prices.shape[0]):
            start, end = idx - length, idx
            xp = np.arange(start, end)
            yp = np.poly1d(np.polyfit(xp, prices[start:end], poly_order))

            curve = yp(xp)
            volatility = np.max(np.abs(curve / prices[start:end] - 1.0))
            direction = curve[-1] / curve[-2] - 1.0
            volatilities[length_idx, idx] = volatility
            directions[length_idx, idx] = direction


if __name__ == '__main__':
    delta = 0.008
    n_degree = 5
    runner_data = pd.read_csv('../tmp/binance_runner.csv')

    n_samples = 19416
    runner_prices = runner_data['price'].to_numpy()[:n_samples]
    runner_deltas = runner_data['delta'].to_numpy()[:n_samples]
    runner_volumes = runner_data['volume'].to_numpy()[:n_samples]
    runner_durations = runner_data['duration'].to_numpy()[:n_samples]

    np.set_printoptions(precision=4)

    #lengths = np.array((5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31, 33, 37,
    #                    39, 43, 47, 51, 57, 63, 69, 75, 83, 91, 100))
    #lengths = np.array((5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31, 33, 37,
    #                    39, 43, 47, 51, 57, 63, 69, 75, 83, 91, 100, 111, 121,
    #                    131, 151, 161, 181, 201))
    lengths = np.arange(7, 60, 2)

    lengths_df = pd.DataFrame(data={'length': lengths})
    lengths_df.to_csv('../tmp/regime_data_lengths.csv')

    spectrum = np.zeros((len(lengths), n_degree * 2, runner_prices.shape[0]))

    for i in range(n_degree):
        make_spectrum(lengths=lengths,
                      prices=runner_prices,
                      poly_order=i + 1,
                      volatilities=spectrum[:, i * 2 + 0, :],
                      directions=spectrum[:, i * 2 + 1, :])

    volatility_factor = (1 / np.power(lengths * 0.0000126, 0.636))[:, None] * 0.3
    direction_factor = pow(lengths * 9.11848707e+02, 5.67883215e-01)[:, None] * 0.3

    for i in range(n_degree):
        spectrum[:, i * 2 + 0, lengths[-1]:] *= volatility_factor
        spectrum[:, i * 2 + 1, lengths[-1]:] *= direction_factor

    out_data = {'price': runner_prices[lengths[-1]:],
                'delta': runner_deltas[lengths[-1]:],
                'volume': runner_volumes[lengths[-1]:],
                'duration': runner_durations[lengths[-1]:]}

    for i in range(n_degree):
        for length_idx, length in enumerate(lengths):
            out_data[f'vol_{i * 2 + 0}_{length_idx}'] = spectrum[length_idx, i * 2 + 0, lengths[-1]:]
            out_data[f'dir_{i * 2 + 1}_{length_idx}'] = spectrum[length_idx, i * 2 + 1, lengths[-1]:]

    spectrum_df = pd.DataFrame(data=out_data)
    spectrum_df.to_csv('../tmp/regime_data.csv')

    fig, axs = plt.subplots(1 + n_degree * 2, 1, sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0})
    axs[0].plot(runner_prices[lengths[-1]:])

    x = np.arange(runner_prices.shape[0] - lengths[-1])
    for i in range(n_degree):
        volatility = 1 - spectrum[:, i * 2 + 0, lengths[-1]:]
        axs[i * 2 + 1].pcolormesh(x, lengths, volatility, vmin=np.min(volatility), vmax=np.max(volatility), shading='auto', cmap=plt.get_cmap('Blues'))
        #axs[1].set_title("Volatility")

        direction = spectrum[:, i * 2 + 1, lengths[-1]:]
        direction_amplitude = np.max(np.abs(direction))
        direction = (direction_amplitude + direction) / (2 * direction_amplitude)
        axs[i * 2 + 2].pcolormesh(x, lengths, direction, vmin=np.min(direction), vmax=np.max(direction), shading='auto', cmap=plt.get_cmap('RdYlGn'))
        #axs[2].set_title("Direction")

    plt.tight_layout()
    plt.show()
    quit()


    for i in range(len(lengths)):
        volatility = spectrum[i, 0, lengths[-1]:] * 1.0 / pow(lengths[i] * 0.0000126, 0.636)
        angle = spectrum[i, 1, lengths[-1]:] * pow(lengths[i] * 9.11848707e+02, 5.67883215e-01)
        #print(np.max(np.abs(angle)))
        axs[0].plot(volatility)
        axs[1].plot(angle)

    plt.show()

    print(length_idx, idx)
    quit()

    print(spectrum)
    quit()



    for length in (5, 7, 9, 11, 13, 15, 19, 23, 27, 33, 39, 47, 57, 69, 83, 101):  # [5, 7, 9, 11, 13]:
        start = runner_prices.shape[0] - length
        end = runner_prices.shape[0]
        x = np.arange(start, end)
        yp = np.poly1d(np.polyfit(x, runner_prices[-length:], 2))
        xp = np.arange(start, end)
        plt.plot(xp, yp(xp))

    plt.show()
    quit()

    lengths = [5, 7, 9, 11, 15, 23, 33, 47, 69, 101, 121, 151, 181, 221, 331]
    for length in lengths:
        savgols.append(savgol_filter(runner_prices, length, 4))

    """
    #plt.plot(xf, np.abs(yf))
    plt.plot(runner_prices)
    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
    plt.show()
    quit()

    b.to_csv('../tmp/savgol.csv')
    print(b)

    quit()
    
    # runner_durations = runner_data['duration'].to_numpy()[0:4000]

    yf = fft(runner_prices[:2048])
    xf = fftfreq(2048, 1)

    #ff = np.abs(np.fft.fft(runner_prices[2000:3024]))[0:512] / 1000 * np.arange(512) ** 0.9
    #print(ff)
    #print(np.arange(512))

    #t = np.linspace(0, 0.5, 500)
    #s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)
    #ff = np.abs(np.fft.fft(s))[:250]

    import matplotlib.pyplot as plt

    #plt.plot(xf, np.abs(yf))
    plt.plot(runner_prices[:2048])
    plt.show()
    quit()
    
    print(ff)
    """
    """
    """

    #smooth_15 = super_smoother(runner_prices, period=65) #[2:]
    #smooth_100 = super_smoother(runner_prices, period=100)[20:]

    plotter = Plotter()

    for idx, ie_price in enumerate(runner_prices):
        plotter.append_event(PositionDirection.hedge, idx, ie_price)
        plotter.append_timestamp(idx, runner_data.iloc[idx]['timestamp'])

    for savgol in savgols:
        plotter.add_indicator(savgol, 'a')
    #plotter.add_indicator(smooth_15, 'Smooth 15')
    #plotter.add_indicator(smooth_100, 'Smooth 100')

    plotter.plot()

    quit()



    x = np.arange(runner_data.shape[0])
    slopes = Slopes(runner_prices, use_cache=False)

    slopes_history_count = 0
    first_idx = Slopes.max_slope_length + slopes_history_count
    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner_prices[0], initial_leverage=0.0)
    plotter = Plotter(slopes=slopes)
    position = Position(delta=delta, direction=PositionDirection.long, plotter=plotter)

    for idx, ie_price in enumerate(runner_prices[:first_idx]):
        plotter.append_event(PositionDirection.hedge, idx, ie_price)
        plotter.append_timestamp(idx, runner_data.iloc[idx]['timestamp'])

    previous_trade_value = simulator.get_value_usdt(mark_price=runner_prices[first_idx])

    previous_make_trade = False

    for idx in range(first_idx, runner_prices.shape[0]):
        slope = slopes.slopes.iloc[idx - Slopes.max_slope_length]
        ie_price = runner_prices[idx]

        plotter.append_slope_length(idx, slope['length'])
        plotter.append_volatility(idx, slope['volatility'])

        previous_duration = runner_durations[idx - 1]
        duration = runner_durations[idx]

        make_trade = position.step(idx, ie_price, runner_prices[idx - 1], duration, previous_duration, slope)

        if make_trade and previous_make_trade:
            if position.direction == PositionDirection.short:
                order_size = simulator.calculate_order_size_btc(leverage=2.5, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            else:
                order_size = simulator.calculate_order_size_btc(leverage=-1.5, mark_price=ie_price)
                simulator.market_order(order_size_btc=order_size, mark_price=ie_price)
            position.direction *= -1
            made_trade = True

            value_after = simulator.get_value_usdt(mark_price=ie_price)
            profit = (value_after - previous_trade_value) / previous_trade_value
            previous_trade_value = value_after

            plotter.append_annotation(x=idx, y=ie_price, direction=PositionDirection.short, profit=profit)

        previous_make_trade = make_trade

        plotter.append_value(idx, simulator.get_value_usdt(mark_price=ie_price))
        plotter.append_angle(idx, slope.angle)
        plotter.append_event(position.direction, idx, ie_price)
        plotter.append_timestamp(idx, runner_data.iloc[idx]['timestamp'])

    plotter.plot()
