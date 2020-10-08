
import math
from enum import Enum

from Common import Direction, RunnerEvent, EventType


class CoastlineRunner:
    def __init__(self, delta_up, delta_down, delta_star_up, delta_star_down):
        self.delta_up = delta_up
        self.delta_down = delta_down
        self.delta_star_up = delta_star_up
        self.delta_star_down = delta_star_down
        self.direction = Direction.up
        self.extreme_price = 0
        self.reference_price = 0
        self.initialized = False

    def step(self, mark_price):
        if not self.initialized:
            self.extreme_price = mark_price.mid
            self.reference_price = mark_price.mid
            self._update_thresholds()
            self.initialized = True

        if self.direction == Direction.up:
            if mark_price.ask <= self.direction_change_threshold:
                self.direction = Direction.down
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

        elif self.direction == Direction.down:
            if mark_price.bid >= self.direction_change_threshold:
                self.direction = Direction.up
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
        if self.direction == Direction.up:
            self.overshoot_threshold = math.exp(math.log(self.extreme_price) + self.delta_star_up)
            self.direction_change_threshold = math.exp(math.log(self.reference_price) - self.delta_down)
        else:
            self.overshoot_threshold = math.exp(math.log(self.extreme_price) - self.delta_star_down)
            self.direction_change_threshold = math.exp(math.log(self.reference_price) + self.delta_up)

    def get_upper_event_type(self):
        if self.direction_change_threshold > self.overshoot_threshold:
            return EventType.direction_change
        else:
            return EventType.overshoot

    def get_lower_event_type(self):
        if self.direction_change_threshold < self.overshoot_threshold:
            return EventType.direction_change
        else:
            return EventType.overshoot

    def get_expected_upper_threshold(self):
        if self.direction_change_threshold > self.overshoot_threshold:
            return self.direction_change_threshold
        else:
            return self.overshoot_threshold

    def get_expected_lower_threshold(self):
        if self.direction_change_threshold < self.overshoot_threshold:
            return self.direction_change_threshold
        else:
            return self.overshoot_threshold
