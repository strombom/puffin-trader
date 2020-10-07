

import sys
sys.path.append("../Common")

from enum import Enum

from CoastlineRunner import CoastlineRunner


class TraderDirection(Enum):
    long = 1
    short = -1


class CoastlineTrader:
    def __init__(self, delta, direction):
        self.delta = delta
        self.direction = direction
        self.runners = []
        self.init_runners()

    def init_runners(self):
        self.runners.append(CoastlineRunner(delta_up=self.delta, delta_down=self.delta, delta_star_up=self.delta, delta_star_down=self.delta))
        if self.direction == TraderDirection.long:
            self.runners.append(CoastlineRunner(delta_up=0.75 * self.delta, delta_down=1.50 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=0.50 * self.delta, delta_down=2.00 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))
        else:
            self.runners.append(CoastlineRunner(delta_up=1.50 * self.delta, delta_down=0.75 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=2.00 * self.delta, delta_down=0.50 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))

