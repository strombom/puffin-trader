import os
from datetime import datetime

import gym
import glob
import torch

import environment
import wandb
import random
import argparse
import numpy as np
from collections import deque
from SAC.agent import SAC
from SAC.utils import save, collect_random
from SAC.buffer import ReplayBuffer


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="SAC" + " " + datetime.now().strftime("%Y-%m-%d_%H%M%S"), help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="T1", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000,
                        help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=1,
                        help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")

    parser.add_argument("--wandb_entity", type=str, default="johan-strombom", help="Wandb entity")
    parser.add_argument("--wandb_api_key", type=str, default="729cdb29cd7a10b4b1cdff9be20b854779840a7b", help="Wandb api key")
    parser.add_argument("--wandb_mode", type=str, default="offline", help="Wandb mode")

    args = parser.parse_args()
    return args


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = environment.TradeEnv()
    #env = gym.make(config.env)
    #env.seed(config.seed)
    #env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    with wandb.init(project="SAC_Discrete", name=config.run_name, config=config, mode=config.wandb_mode):

        agent = SAC(state_size=env.observation_space.shape[0],
                    action_size=env.action_space.n,
                    device=device)

        wandb.watch(agent, log="gradients", log_freq=10)
        buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)
        collect_random(env=env, dataset=buffer, num_samples=10000)

        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x % 1 == 0, force=True)

        for i in range(1, config.episodes + 1):
            state = env.reset()
            episode_steps = 0
            rewards = 0
            while True:
                action = agent.get_action(state)
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)
                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, current_alpha = agent.learn(steps,
                                                                                                       buffer.sample(),
                                                                                                       gamma=0.99)
                state = next_state
                rewards += reward
                episode_steps += 1
                if done:
                    break

            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

            wandb.log({"Reward": rewards,
                       "Average10": np.mean(average10),
                       "Steps": total_steps,
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Bellmann error 1": bellmann_error1,
                       "Bellmann error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Steps": steps,
                       "Episode": i,
                       "Buffer size": buffer.__len__()})

            if (i % 10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: ' + str(i - 10), fps=4, format="gif"),
                               "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="SAC_discrete", model=agent.actor_local, wandb=wandb, ep=0)


if __name__ == "__main__":
    config = get_config()
    train(config)
