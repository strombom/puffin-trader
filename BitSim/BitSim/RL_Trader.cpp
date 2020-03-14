#include "pch.h"

#include "RL_Trader.h"
#include "BitBotConstants.h"


void RL_Trader::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Closer::n_episodes; ++idx_episode) {
        auto state = environment.reset();
        step_episode = 0;

        while (!state.is_done()) {
            auto action = get_action(state);
            state = environment.step(action);

            ++step_total;
            ++step_episode;
        }

        auto reward = environment.get_reward();

        update_model();

        if (idx_episode % BitSim::Closer::save_period == 0 ||
            idx_episode == BitSim::Closer::n_episodes - 1) {
            save_params(idx_episode);
            interim_test();
        }
    }
}

void RL_Trader::update_model(void)
{

}

void RL_Trader::save_params(int idx_period)
{

}

void RL_Trader::interim_test(void)
{

}

RL_Action RL_Trader::get_action(RL_State state)
{
    if (step_total < BitSim::Closer::initial_random_action) {
        //return actor.get_random_action();
    }

    return actor.get_action(state);
}

RL_State RL_Trader::step(RL_Action action)
{
    auto next_state = environment.step(action);
    
    // TODO: Add transition to memory

    return next_state;
}
