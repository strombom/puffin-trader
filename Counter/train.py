
import gym
import time
import torch
import numpy as np
from SAC_old.agent import SAC
from SAC_old.logger import Logger
from SAC_old.utils import stack_states


class Play:
    def __init__(self, env, agent, params, max_episode=4):
        self.env = env
        self.params = params
        self.env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_to_eval_mode()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):
        stacked_states = np.zeros(shape=self.params["state_shape"], dtype=np.uint8)
        total_reward = 0
        print("--------Play mode--------")
        for _ in range(self.max_episode):
            done = 0
            state = self.env.reset()
            episode_reward = 0
            stacked_states = stack_states(stacked_states, state, True)

            while not done:
                stacked_frames_copy = stacked_states.copy()
                action = self.agent.choose_action(stacked_frames_copy, do_greedy=False)
                next_state, r, done, _ = self.env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                self.env.render()
                time.sleep(0.01)
                episode_reward += r
            total_reward += episode_reward

        print("Total episode reward:", total_reward / self.max_episode)
        self.env.close()
        #cv2.destroyAllWindows()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    config = {
        "lr": 3e-4,
        "batch_size": 64,
        "state_shape": (4, ),
        "max_steps": int(1e+8),
        "gamma": 0.99,
        "initial_random_steps": 200,
        "train_period": 4,
        "fixed_network_update_freq": 8000,
        "mem_size": 1000,
        "env_name": "CartPole-v1",
        "interval": 4,
        "do_train": True,
        "train_from_scratch": True
    }

    env = gym.make(config['env_name'])
    config['n_actions'] = env.action_space.n

    agent = SAC(**config)
    logger = Logger(agent, **config)

    if config['do_train']:
        if config['train_from_scratch']:
            min_episode = 0
        else:
            episode = logger.load_weights()
            agent.hard_update_target_network()
            agent.alpha = agent.log_alpha.exp()
            min_episode = episode

        stacked_states = np.zeros(shape=config["state_shape"], dtype=np.uint8)
        state = env.reset()
        stacked_states = stack_states(stacked_states, state, True)
        episode_reward = 0
        alpha_loss, q_loss, policy_loss = 0, 0, 0
        episode = min_episode + 1
        logger.on()
        for step in range(1, config["max_steps"] + 1):
            if step < config['initial_random_steps']:
                stacked_states_copy = stacked_states.copy()
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                reward = np.sign(reward)
                agent.store(stacked_states_copy, action, reward, stacked_states, done)
                if done:
                    state = env.reset()
                    stacked_states = stack_states(stacked_states, state, True)
            else:
                stacked_states_copy = stacked_states.copy()
                action = agent.choose_action(stacked_states_copy)
                next_state, reward, done, _ = env.step(action)
                stacked_states = stack_states(stacked_states, next_state, False)
                reward = np.sign(reward)
                agent.store(stacked_states_copy, action, reward, stacked_states, done)
                episode_reward += reward
                state = next_state

                if step % config["train_period"] == 0:
                    alpha_loss, q_loss, policy_loss = agent.train()

                if done:
                    logger.off()
                    logger.log(episode, episode_reward, alpha_loss, q_loss, policy_loss , step)

                    episode += 1
                    obs = env.reset()
                    state = stack_states(state, obs, True)
                    episode_reward = 0
                    episode_loss = 0
                    logger.on()

    logger.load_weights()
    player = Play(env, agent, config)
    player.evaluate()
