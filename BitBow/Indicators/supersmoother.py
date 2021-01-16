
# import sys
# sys.path.insert(0, '../')
# sys.path.insert(1, '../../')

import math
# import numpy

# from plot import plotter
# from database.token_db import TokenDB


class SuperSmoother:
    def __init__(self, period, initial_value = 0):
        a = math.e ** (-1.414 * 3.14159 / period)
        b = 2 * a * math.cos(1.414 * 3.14159 / period)
        self.c2 = b
        self.c3 = -a * a
        self.c1 = 1 - self.c2 - self.c3
        self.input_1 = initial_value
        self.output_1 = initial_value
        self.output_2 = initial_value

    def append(self, data):
        output = self.c1 * (data + self.input_1) / 2
        output += self.c2 * self.output_1 + self.c3 * self.output_2
        self.input_1 = data
        self.output_2 = self.output_1
        self.output_1 = output
        return output

"""
if __name__ == '__main__':
    token_db = TokenDB('../../database/token_btc_usdt.db')

    data_length = 20000

    prices = token_db.get_prices_volume(data_length, aggregate_rows = 1)

    smooths = [(prices['high'], 'Price high')]

    for time_constant in [10, 16, 25, 40, 63,
                          100, 160, 250, 400, 630,
                          1000, 1600, 2500, 4000, 6300,
                          10000, 20000, 50000]:
        super_smoother = SuperSmoother(time_constant, prices['high'][0])
        smooth = numpy.zeros(data_length)
        for idx, price in enumerate(prices['high']):
            out = super_smoother.append(price)
            smooth[idx] = out
        smooths.append((smooth, "Smooth " + str(time_constant)))

    print(smooths)
"""
