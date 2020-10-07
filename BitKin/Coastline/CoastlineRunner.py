
from enum import Enum
import math


class RunnerDirection(Enum):
    up = 1
    down = 2


class RunnerEvent(Enum):
    nothing = 0
    direction_change_up = 1
    direction_change_down = -1
    overshoot_up = 2
    overshoot_down = 3


class CoastlineRunner:
    def __init__(self, delta_up, delta_down, delta_star_up, delta_star_down):
        self.delta_up = delta_up
        self.delta_down = delta_down
        self.delta_star_up = delta_star_up
        self.delta_star_down = delta_star_down
        self.direction = RunnerDirection.up
        self.extreme_price = 0
        self.reference_price = 0
        self.initialized = False

    def step(self, mark_price):
        if not self.initialized:
            self.extreme_price = mark_price.mid
            self.reference_price = mark_price.mid
            self._update_thresholds()
            self.initialized = True

        if self.direction == RunnerDirection.up:
            if mark_price.ask <= self.direction_change_threshold:
                self.direction = RunnerDirection.down
                self.extreme_price = mark_price.ask
                self.reference_price = mark_price.ask
                self._update_thresholds()
                return RunnerEvent.direction_change_down

            if mark_price.bid > self.extreme_price:
                self.extreme_price = mark_price.bid
                self._update_thresholds()
                if mark_price.bid > self.overshoot_threshold:
                    self.reference_price = self.extreme_price
                    self._update_thresholds()
                    return RunnerEvent.overshoot_down

        elif self.direction == RunnerDirection.down:
            if mark_price.bid >= self.direction_change_threshold:
                self.direction = RunnerDirection.up
                self.extreme_price = mark_price.bid
                self.reference_price = mark_price.bid
                self._update_thresholds()
                return RunnerEvent.direction_change_up

            if mark_price.ask < self.extreme_price:
                self.extreme_price = mark_price.ask
                self._update_thresholds()
                if mark_price.ask < self.overshoot_threshold:
                    self.reference_price = mark_price.ask
                    self.extreme_price = mark_price.ask
                    self._update_thresholds()
                    return RunnerEvent.overshoot_up

        return RunnerEvent.nothing

    def _update_thresholds(self):
        if self.direction == RunnerDirection.up:
            self.overshoot_threshold = math.exp(math.log(self.extreme_price) + self.delta_star_up)
            self.direction_change_threshold = math.exp(math.log(self.reference_price) - self.delta_down)
        else:
            self.overshoot_threshold = math.exp(math.log(self.extreme_price) - self.delta_star_down)
            self.direction_change_threshold = math.exp(math.log(self.reference_price) + self.delta_up)

