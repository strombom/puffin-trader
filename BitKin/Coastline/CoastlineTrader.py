
from CoastlineRunner import CoastlineRunner
from Common import OrderSide


class CoastlineTrader:
    def __init__(self, delta, direction):
        self.delta = delta
        self.unit_size = 1.0
        self.direction = direction
        self.runners = []
        self.init_runners()

    def init_runners(self):
        self.runners.append(CoastlineRunner(delta_up=self.delta, delta_down=self.delta, delta_star_up=self.delta, delta_star_down=self.delta))
        if self.direction == OrderSide.long:
            self.runners.append(CoastlineRunner(delta_up=0.75 * self.delta, delta_down=1.50 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=0.50 * self.delta, delta_down=2.00 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))
        else:
            self.runners.append(CoastlineRunner(delta_up=1.50 * self.delta, delta_down=0.75 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=2.00 * self.delta, delta_down=0.50 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))

    def step(self, mark_price):
        print("step", mark_price.ask, mark_price.mid, mark_price.bid)

        events = []
        for runner in self.runners:
            events.append(runner.step(mark_price))

