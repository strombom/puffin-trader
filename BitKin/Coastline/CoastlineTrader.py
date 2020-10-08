
from CoastlineRunner import CoastlineRunner
from Liquidity import Liquidity
from LimitOrder import LimitOrder, LimitOrderLevel
from Common import Direction, OrderSide, EventType


class CoastlineTrader:
    def __init__(self, delta, direction):
        self.delta = delta
        self.reference_unit_size = 1.0
        self.current_unit_size = self.reference_unit_size
        self.inventory = 0.0
        self.direction = direction
        self.liquidity = Liquidity(delta, delta * 2.525729, 50.0)
        self.unbalanced_filled_orders = []
        self.sell_order = None
        self.buy_order = None
        self.runners = []
        self.init_runners()
        self.initialized = False

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

        self.liquidity.update(mark_price)

        events = []
        for runner in self.runners:
            events.append(runner.step(mark_price))

        if not self.initialized:
            self.update_unit_size()
            self.put_orders(mark_price)
            self.initialized = True
            return

    def put_orders(self, mark_price):
        if self.liquidity.get_liquidity() >= 0.5:
            cascade_coef = 1.0
        elif self.liquidity.get_liquidity() >= 0.1:
            cascade_coef = 0.5
        else:
            cascade_coef = 0.1

        cascade_volume = self.current_unit_size * cascade_coef

        runner = self.select_current_runner()

        if runner.direction == Direction.up:
            buy_event_type = EventType.direction_change
            buy_delta = runner.delta_down
            sell_event_type = EventType.overshoot
            sell_delta = runner.delta_star_up
        else:
            buy_event_type = EventType.overshoot
            buy_delta = runner.delta_star_down
            sell_event_type = EventType.direction_change
            sell_delta = runner.delta_up

        if self.direction == Direction.up:
            if len(self.unbalanced_filled_orders) == 0:
                self.sell_order = None
                self.buy_order = LimitOrder(side=OrderSide.long, price=mark_price, volume=cascade_volume, event_type=runner.get_lower_event_type())
            else:
                self.buy_order = LimitOrder(side=OrderSide.long, price=mark_price, volume=cascade_volume, event_type=runner.get_upper_event_type())
                balanced_filled_orders = self.find_balanced_filled_orders(runner.get_expected_upper_threshold(), Direction.down)
                if len(balanced_filled_orders) > 0:
                    self.sell_order = LimitOrder(side=OrderSide.short, price=mark_price, volume=0, event_type=sell_event_type)
                    self.sell_order.balance_orders(balanced_filled_orders)
                else:
                    self.sell_order = None


    def select_current_runner(self):
        if abs(self.inventory) < 15:
            return self.runners[0]
        elif abs(self.inventory) < 30:
            return self.runners[1]
        else:
            return self.runners[2]

    def update_unit_size(self):
        if abs(self.inventory) < 15:
            self.current_unit_size = self.reference_unit_size
        elif abs(self.inventory) < 30:
            self.current_unit_size = self.reference_unit_size * 0.5
        else:
            self.current_unit_size = self.reference_unit_size * 0.25

    def find_balanced_filled_orders(self, threshold, direction):
        return []
