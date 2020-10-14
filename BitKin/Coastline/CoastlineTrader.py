
import sys
sys.path.append("../Common")

from BitmexSimulator import Position
from CoastlineRunner import CoastlineRunner
from Liquidity import Liquidity
from LimitOrder import LimitOrder
from Common import Direction, OrderSide, EventType, RunnerEvent


class Logger:
    def __init__(self, simulator):
        self.simulator = simulator
        self.file = open('trades.csv', 'w')

    def order(self, order_type, price, volume):
        print(f'log {order_type} {volume} @ {price} c:{self.simulator.contracts} w:{self.simulator.wallet}')
        self.file.write(f'{order_type},{volume},{price},{self.simulator.contracts},{self.simulator.wallet}\n')

    def event(self, event_idx, event, mark_price, selected):
        print(f'log event {event_idx} {event} {mark_price} {selected}')
        self.file.write(f'{event},{event_idx},{mark_price.ask},{mark_price.bid},{selected}\n')


class CoastlineTrader:
    def __init__(self, delta, order_side, initial_price):
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
        self.position_realized_profit = 0.0
        self.runners = []
        self.init_runners()
        self.initialized = False
        self.bitmex_simulator = Position({'max_leverage': 10}, initial_price)
        self.logger = Logger(self.bitmex_simulator)

    def init_runners(self):
        self.runners.append(CoastlineRunner(delta_up=self.delta, delta_down=self.delta, delta_star_up=self.delta, delta_star_down=self.delta))
        if self.order_side == OrderSide.long:
            self.runners.append(CoastlineRunner(delta_up=0.75 * self.delta, delta_down=1.50 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=0.50 * self.delta, delta_down=2.00 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))
        else:
            self.runners.append(CoastlineRunner(delta_up=1.50 * self.delta, delta_down=0.75 * self.delta, delta_star_up=0.75 * self.delta, delta_star_down=0.75 * self.delta))
            self.runners.append(CoastlineRunner(delta_up=2.00 * self.delta, delta_down=0.50 * self.delta, delta_star_up=0.50 * self.delta, delta_star_down=0.50 * self.delta))

    def step(self, mark_price):
        #print("step", mark_price.ask, mark_price.mid, mark_price.bid)

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

        for idx, event in enumerate(events):
            if idx == runner_idx:
                continue
            if event != RunnerEvent.nothing:
                self.logger.event(idx, event, mark_price, False)

        if events[runner_idx] != RunnerEvent.nothing:
            self.logger.event(runner_idx, events[runner_idx], mark_price, True)
            self.buy_order, self.sell_order = None, None
            self.put_orders(mark_price)
            return

        target_abs_pnl = 10.0
        if self.get_upnl(mark_price) + self.position_realized_profit >= target_abs_pnl:
            self.close_position(mark_price)
            self.put_orders(mark_price)
        else:
            if self.buy_order is not None:
                self.balance_buy_order(runner.direction_change_threshold)
            if self.sell_order is not None:
                self.balance_sell_order(runner.direction_change_threshold)

    def balance_buy_order(self, direction_change_threshold):
        if self.buy_order.side != OrderSide.long or direction_change_threshold <= self.buy_order.price or \
                (self.order_side == OrderSide.short and len(self.unbalanced_filled_orders) == 0):
            return

        if self.order_side == OrderSide.short and len(self.unbalanced_filled_orders) > 1:
            balanced_orders = self.find_balanced_orders(direction_change_threshold, Direction.up)
            if len(balanced_orders) == 0:
                self.buy_order = None
            else:
                self.buy_order.price = round(direction_change_threshold * 2) / 2
                self.buy_order.set_balanced_orders(balanced_orders)
        else:
            self.buy_order.price = round(direction_change_threshold * 2) / 2

    def balance_sell_order(self, direction_change_threshold):
        if self.sell_order.side != OrderSide.short or direction_change_threshold >= self.sell_order.price or \
                (self.order_side == OrderSide.long and len(self.unbalanced_filled_orders) == 0):
            return

        if self.order_side == OrderSide.long or len(self.unbalanced_filled_orders) > 1:
            balanced_orders = self.find_balanced_orders(direction_change_threshold, Direction.down)
            if len(balanced_orders) == 0:
                self.sell_order = None
            else:
                self.sell_order.price = round(direction_change_threshold * 2) / 2
                self.sell_order.set_balanced_orders(balanced_orders)
        else:
            self.sell_order.price = round(direction_change_threshold * 2) / 2

    def put_orders(self, mark_price):
        liquidity = self.liquidity.get_liquidity()
        if liquidity >= 0.5:
            cascade_coef = 1.0
        elif liquidity >= 0.1:
            cascade_coef = 0.5
        else:
            cascade_coef = 0.1

        cascade_volume = self.current_unit_size * cascade_coef

        runner = self.runners[self.select_current_runner()]

        if runner.direction == Direction.up:
            buy_event_type = EventType.direction_change
            sell_event_type = EventType.overshoot
        else:
            buy_event_type = EventType.overshoot
            sell_event_type = EventType.direction_change

        if self.order_side == OrderSide.long:
            if len(self.unbalanced_filled_orders) == 0:
                self.sell_order = None
                self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=cascade_volume, event_type=runner.get_lower_event_type())
            else:
                self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=cascade_volume, event_type=buy_event_type)
                balanced_orders = self.find_balanced_orders(runner.get_expected_upper_threshold(), Direction.down)
                if len(balanced_orders) == 0:
                    self.sell_order = None
                else:
                    self.sell_order = LimitOrder(side=OrderSide.short, price=runner.get_expected_upper_threshold(), volume=0, event_type=sell_event_type)
                    self.sell_order.set_balanced_orders(balanced_orders)
        else:
            if len(self.unbalanced_filled_orders) == 0:
                self.buy_order = None
                self.sell_order = LimitOrder(side=OrderSide.short, price=runner.get_expected_upper_threshold(), volume=cascade_volume, event_type=runner.get_upper_event_type())
            else:
                balanced_orders = self.find_balanced_orders(runner.get_expected_lower_threshold(), Direction.down)
                if len(balanced_orders) == 0:
                    self.buy_order = None
                else:
                    self.buy_order = LimitOrder(side=OrderSide.long, price=runner.get_expected_lower_threshold(), volume=0, event_type=buy_event_type)
                    self.buy_order.set_balanced_orders(balanced_orders)
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
            self.bitmex_simulator.limit_order(order_contracts=self.buy_order.volume * mark_price.ask, mark_price=mark_price.ask)
            self.logger.order('limit_buy', mark_price.ask, self.buy_order.volume)
            self.inventory += self.buy_order.volume
            self.update_unit_size()
            if self.order_side == OrderSide.long:
                self.unbalanced_filled_orders.append(self.buy_order)
            else:
                self.position_realized_profit += self.buy_order.get_relative_pnl()
                for balanced_order in self.buy_order.balanced_orders:
                    if balanced_order in self.unbalanced_filled_orders:
                        self.unbalanced_filled_orders.remove(balanced_order)

            if len(self.unbalanced_filled_orders) == 0:
                self.close_position(mark_price)

            self.buy_order = None
            self.sell_order = None

    def evaluate_sell_orders(self, mark_price):
        if self.sell_order is not None and mark_price.bid > self.sell_order.price:
            self.bitmex_simulator.limit_order(order_contracts=-self.sell_order.volume * mark_price.bid, mark_price=mark_price.bid)
            self.logger.order('limit_sell', mark_price.bid, self.sell_order.volume)
            self.inventory -= self.sell_order.volume
            self.update_unit_size()
            if self.order_side == OrderSide.short:
                self.unbalanced_filled_orders.append(self.sell_order)
            else:
                self.position_realized_profit += self.sell_order.get_relative_pnl()
                for balanced_order in self.sell_order.balanced_orders:
                    if balanced_order in self.unbalanced_filled_orders:
                        self.unbalanced_filled_orders.remove(balanced_order)

            if len(self.unbalanced_filled_orders) == 0:
                self.close_position(mark_price)

            self.sell_order = None
            self.buy_order = None

    def get_upnl(self, mark_price):
        if len(self.unbalanced_filled_orders) == 0:
            return 0.0

        if self.order_side == OrderSide.long:
            market_price = mark_price.bid
        else:
            market_price = mark_price.ask

        upnl = 0.0
        for order in self.unbalanced_filled_orders:
            if order.side == OrderSide.long:
                price_move = market_price - order.price
            else:
                price_move = order.price - market_price
            upnl += price_move / order.price * order.volume

        return upnl

    def close_position(self, mark_price):
        # Close positions with market order
        if self.buy_order is not None:
            self.bitmex_simulator.market_order(order_contracts=self.buy_order.volume * mark_price.bid, mark_price=mark_price.bid)
            self.logger.order('market_buy', mark_price.ask, self.buy_order.volume)
        if self.sell_order is not None:
            self.bitmex_simulator.market_order(order_contracts=-self.buy_order.volume * mark_price.bid, mark_price=mark_price.bid)
            self.logger.order('market_sell', mark_price.bid, self.buy_order.volume)

        self.realized_profit += self.get_upnl(mark_price)
        self.realized_profit += self.position_realized_profit
        self.position_realized_profit = 0.0
        self.unbalanced_filled_orders = []
        self.inventory = 0.0
        self.buy_order = None
        self.sell_order = None
        self.update_unit_size()
