import pickle
import random


class BitmexSim:
    def __init__(self, max_steps):
        with open(f"cache/intrinsic_time_data.pickle", 'rb') as f:
            deltas, order_books, runners, runner_clock, clock_TMV, clock_R = pickle.load(f)

        self.deltas = deltas
        self.order_books = order_books
        self.runners = runners
        self.runner_clock = runner_clock
        self.clock_TMV = clock_TMV
        self.clock_R = clock_R

        self.max_steps = max_steps
        self.step_count = 0
        self.step_idx = 0
        self.reset()

        # Actions:
        # 0 - Nop
        # 1 - Market buy
        # 2 - Market sell

    def seed(self, seed):
        pass

    def step(self, action):

        print('deltas', len(self.deltas))
        print('order_books', len(self.order_books))
        print('runner_clock.ie_prices', len(self.runner_clock.ie_prices))
        print('runners[0].ie_prices', len(self.runners[0].ie_prices))
        print('runners[-1].ie_prices', len(self.runners[-1].ie_prices))
        print('clock_TMV', self.clock_TMV.shape)
        print('clock_R', self.clock_R.shape)

        self.step_idx += 1
        done = self.step_idx == len(self.runner_clock.ie_prices)

        observation, reward = [], 0.0
        return observation, reward, done

    def reset(self):
        self.step_count = 0
        self.step_idx = random.randint(0, len(self.runner_clock.ie_prices) - self.max_steps - 1)

        observation = []
        return self.step()[0]

    def render(self):
        pass

    def close(self):
        pass
