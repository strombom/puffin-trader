
import pandas as pd
from tsai.all import *

from BinanceSim.binance_simulator import BinanceSimulator


if __name__ == '__main__':
    n_timesteps = 10
    regime_predictor = load_learner_all('regime_model')

    lengths = pd.read_csv('../tmp/regime_data_lengths.csv')
    regime_data = pd.read_csv('../tmp/regime_data.csv')

    regime_data['duration'] /= 100000000
    regime_data['volume'] /= 2000
    regime_data['delta'] /= 0.008

    regime_data_raw = regime_data.to_numpy()[:, 2:]
    print(regime_data_raw.shape)

    runner_prices = regime_data['price'].to_numpy()
    print(runner_prices)

    simulator = BinanceSimulator(initial_usdt=0.0, initial_btc=1.0, max_leverage=2, mark_price=runner_prices[n_timesteps - 2], initial_leverage=0.0)
    direction = 1

    prices = []
    values = []


    for idx in range(n_timesteps - 1, runner_prices.shape[0]):
        features = regime_data_raw[idx - n_timesteps + 1:idx + 1, :].transpose()
        print(features.shape)
        quit()
        prediction = regime_predictor.get_X_preds(features)[0][0]

        mark_price = runner_prices[idx]

        if prediction[0] > 0.004 and direction == -1:
            order_size = simulator.calculate_order_size_btc(leverage=2.5, mark_price=mark_price)
            simulator.market_order(order_size_btc=order_size, mark_price=mark_price)
            direction = 1
        elif prediction[0] < -0.004 and direction == 1:
            order_size = simulator.calculate_order_size_btc(leverage=-1.5, mark_price=mark_price)
            simulator.market_order(order_size_btc=order_size, mark_price=mark_price)
            direction = -1

        value = simulator.get_value_usdt(mark_price=mark_price)
        prices.append(mark_price)
        values.append(value)
        print("p:", prediction[0:4], mark_price, value)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(prices)
    axs[1].plot(values)
    plt.show()