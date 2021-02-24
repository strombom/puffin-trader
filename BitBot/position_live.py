
from slopes import Slope, Slopes
from plotter import Plotter
from Common.Misc import PositionDirection, Regime


class PositionLive:
    def __init__(self, delta: float, initial_price: float, direction: PositionDirection):
        self.delta = delta
        self.direction = direction
        self.regime = Regime.chop
        self.mark_price_prev = initial_price
        self.prev_slope_angle = 0

    def step(self, mark_price: float, slope: Slope) -> bool:
        # regime
        if self.regime == Regime.chop:
            if (slope.angle > 0.25 and mark_price > self.mark_price_prev) or \
                    (slope.angle < -0.25 and mark_price < self.mark_price_prev):
                if slope.length > 0.20 and slope.volatility > 0.3 - 0.2 * slope.length:
                    self.regime = Regime.trend
                    # self.plotter.regime_change(x=ie_idx, mark_price=mark_price, regime=self.regime)

        # threshold_delta = (1.6 + 0.8 * (slope_len - 10) / 70) * delta
        threshold_delta = 1.85 * self.delta

        if abs(slope.angle) > 2 and slope.length > 0.4:
            threshold_delta *= 0.9
        # elif abs(slope_angle) > 1:
        #     threshold_delta *= 1.1

        # threshold_delta *= (1 - max(0.5, min(1, abs(anglediff))) / 2)

        make_trade = False

        angle_threshold = 0.2 / 5

        if self.direction == PositionDirection.short:
            threshold = slope.y[-1] * (1 + threshold_delta)
            # self.plotter.append_threshold(ie_idx, threshold)

            if mark_price > threshold or \
                    (mark_price > slope.y[-1] and slope.angle > angle_threshold and slope.angle > self.prev_slope_angle and mark_price > self.mark_price_prev) or \
                    (mark_price > slope.y[-1] and slope.angle > 0 and slope.length > 0.3 and mark_price > self.mark_price_prev):
                make_trade = True

        elif self.direction == PositionDirection.long:
            threshold = slope.y[-1] * (1 - threshold_delta)
            # self.plotter.append_threshold(ie_idx, threshold)

            if mark_price < threshold or \
                    (mark_price < slope.y[-1] and slope.angle < -angle_threshold and slope.angle < self.prev_slope_angle and mark_price < self.mark_price_prev) or \
                    (mark_price < slope.y[-1] and slope.angle < 0 and slope.length > 0.3 and mark_price < self.mark_price_prev):
                make_trade = True

        self.prev_slope_angle = slope.angle
        # if abs(anglediff) > 1.0:
        #     threshold_delta *= 1 - 0.5
        # elif abs(anglediff) > 0.7:
        #     threshold_delta *= 1 - 0.3
        # elif abs(anglediff) > 0.4:
        #     threshold_delta *= 1 - 0.2

        if make_trade and self.regime == Regime.trend:
            self.regime = Regime.chop
            # self.plotter.regime_change(x=ie_idx, mark_price=mark_price, regime=self.regime)

        return make_trade
