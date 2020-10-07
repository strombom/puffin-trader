
import math

from Common import Direction
from CoastlineRunner import RunnerEvent


class Liquidity:
    def __init__(self, delta, delta_star, alpha):
        self.direction = Direction.up
        self.delta_up = delta
        self.delta_down = delta
        self.delta_star = delta_star
        self.alpha = alpha
        self.alpha_weight = math.exp(-2.0 / (alpha + 1.0))
        self.h1 = -math.exp(-delta_star / delta) * math.log(math.exp(-delta_star / delta)) \
                  - (1.0 - math.exp(-delta_star / delta)) * math.log(1.0 - math.exp(-delta_star / delta))
        self.h2 = math.exp(-delta_star / delta)*math.pow(math.log(math.exp(-delta_star / delta)), 2.0) \
                  - (1.0 - math.exp(-delta_star / delta)) * math.pow(math.log(1.0 - math.exp(-delta_star / delta)), 2.0) \
                  - self.h1 * self.h1
        self.liquidity = 0.0
        self.surprise = 0.0

    def get_liquidity(self):
        return self.liquidity

    def update(self, mark_price):
        event = self.step(mark_price)
        if event != RunnerEvent.nothing:
            if event == RunnerEvent.direction_change_up or event == RunnerEvent.direction_change_down:
                k = 0.08338161
            else:
                k = 2.525729
            self.surprise = self.alpha_weight * k + (1.0 - self.alpha_weight) * self.surprise
            self.liquidity = 1.0 - self.cumnorm(math.sqrt(self.alpha) * (self.surprise - self.h1) / math.sqrt(self.h2))

    def step(self, mark_price):
        return 0.0

    def cumnorm(self, x):
        if x > 6.0:
            return 1.0
        if x < -6.0:
            return 0.0
        b1 = 0.31938153
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        p = 0.2316419
        c2 = 0.3989423
        a = abs(x)
        t = 1.0 / (1.0 + a * p)
        b = c2 * math.exp((-x) * (x / 2.0))
        n = ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
        n = 1.0 - b * n
        if x < 0.0:
            n = 1.0 - n
        return n
