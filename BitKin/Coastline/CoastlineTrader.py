
from CoastlineRunner import CoastlineRunner
from Liquidity import Liquidity
from LimitOrder import LimitOrder
from Common import Direction, OrderSide, EventType, RunnerEvent


class CoastlineTrader:
    def __init__(self, delta, order_side):
        self.delta = delta
        self.reference_unit_size = 1.0
        self.current_unit_size = self.reference_unit_size
        self.inventory = 0.0
        self.order_side = order_side
        self.liquidity = Liquidity(delta, delta * 2.525729, 50.0)
        self.unbalanced_filled_orders = []
        self.sell_order = None
        self.buy_order = None
        self.realized_profit = 0.0
        self.runners = []
        self.init_runners()
        self.initialized = False

    def init_runners(self):
        self.runners.append(CoastlineRunner(delta_up=self.delta, delta_down=self.delta, delta_star_up=self.delta, delta_star_down=self.delta))
        if self.order_side == OrderSide.long:
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

        self.evaluate_buy_orders(mark_price)
        self.evaluate_sell_orders(mark_price)

        runner_idx = self.select_current_runner()
        runner = self.runners[runner_idx]

        if events[runner_idx] != RunnerEvent.nothing:
            self.buy_order, self.sell_order = None, None
            self.put_orders(mark_price)
            return

        target_abs_pnl = 10.0
        if self.get_pnl(mark_price) >= target_abs_pnl:
            self.close_position(mark_price)
            self.put_orders(mark_price)
        else:
            if self.buy_order is not None:
                self.correct_buy_order(runner.direction_change_threshold)
            if self.buy_order is not None:
                self.correct_sell_order(runner.direction_change_threshold)

    def get_pnl(self, mark_price):
        upnl = 0
        if len(self.unbalanced_filled_orders) > 0:
            if self.order_side == OrderSide.long:
                market_price = mark_price.bid
            else:
                market_price = mark_price.ask

            for order in self.unbalanced_filled_orders:
                if order.side == OrderSide.long:
                    price_move = market_price - order.price
                else:
                    price_move = order.price - market_price
                upnl = price_move / order.price * order.volume

        return self.realized_profit + upnl

    def put_orders(self, mark_price):
        if self.liquidity.get_liquidity() >= 0.5:
            cascade_coef = 1.0
        elif self.liquidity.get_liquidity() >= 0.1:
            cascade_coef = 0.5
        else:
            cascade_coef = 0.1

        cascade_volume = self.current_unit_size * cascade_coef

        runner = self.runners[self.select_current_runner()]

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

        if self.order_side == OrderSide.long:
            if len(self.unbalanced_filled_orders) == 0:
                self.sell_order = None
                self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=cascade_volume, event_type=runner.get_lower_event_type())
            else:
                self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=cascade_volume, event_type=buy_event_type)
                balanced_orders = self.find_balanced_orders(runner.get_expected_upper_threshold(), Direction.down)
                if len(balanced_orders) > 0:
                    self.sell_order = LimitOrder(side=OrderSide.short, price=runner.get_expected_upper_threshold(), volume=0, event_type=sell_event_type)
                    self.sell_order.balance_orders(balanced_orders)
                else:
                    self.sell_order = None
        else:
            if len(self.unbalanced_filled_orders) == 0:
                self.buy_order = None
                self.sell_order = LimitOrder(side=OrderSide.short, price=runner.get_expected_upper_threshold(), volume=cascade_volume, event_type=runner.get_upper_event_type())
            else:
                balanced_orders = self.find_balanced_orders(runner.get_expected_lower_threshold(), Direction.down)
                if len(balanced_orders) > 0:
                    self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=0, event_type=buy_event_type)
                    self.buy_order.balance_orders(balanced_orders)
                else:
                    self.buy_order = None
                self.sell_order = LimitOrder(side=OrderSide.short, price=runner.get_expected_upper_threshold(), volume=cascade_volume, event_type=sell_event_type)

    def select_current_runner(self):
        if abs(self.inventory) < 15:
            return 0
        elif abs(self.inventory) < 30:
            return 1
        else:
            return 2

    def update_unit_size(self):
        if abs(self.inventory) < 15:
            self.current_unit_size = self.reference_unit_size
        elif abs(self.inventory) < 30:
            self.current_unit_size = self.reference_unit_size * 0.5
        else:
            self.current_unit_size = self.reference_unit_size * 0.25

    def find_balanced_orders(self, threshold, direction):
        balanced_orders = []
        for order in self.unbalanced_filled_orders:
            if (direction == Direction.up and order.price - threshold >= self.delta * order.price) or \
               (direction == Direction.down and threshold - order.price >= self.delta * order.price):
                balanced_orders.append(order)
        return balanced_orders

    def evaluate_buy_orders(self, mark_price):
        if self.buy_order is not None and mark_price.ask < self.buy_order.price:
            self.inventory += self.buy_order.volume
            self.update_unit_size()
            if self.order_side == OrderSide.long:
                self.unbalanced_filled_orders.append(self.buy_order)
            else:
                self.realized_profit += self.buy_order.get_pnl()
                print("a")

            if len(self.unbalanced_filled_orders) == 0:
                self.close_position(mark_price)

            self.buy_order = None
            #makeBuyFilled(price);
            #cancelSellLimitOrder
            self.sell_order = None

    def evaluate_sell_orders(self, mark_price):
        if self.sell_order is not None and mark_price.bid > self.buy_order.price:
            pass
        pass

    def close_position(self, mark_price):
        pass

    def correct_buy_order(self, direction_change_threshold):
        pass

    def correct_sell_order(self, direction_change_threshold):
        pass
