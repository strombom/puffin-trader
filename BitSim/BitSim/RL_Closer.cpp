#include "pch.h"

#include "RL_Closer.h"
#include "BitBotConstants.h"


void RL_Closer::train(void)
{
    for (auto idx_episode = 0; idx_episode < BitSim::Closer::n_episodes; ++idx_episode) {
        auto state = environment.reset();
        step_episode = 0;

        while (true) {
            auto action = get_action(state);
            auto [next_state, done] = step(action);

            ++step_total;
            ++step_episode;

            if (done) {
                break;
            }
        }

        update_model();

        if (idx_episode % BitSim::Closer::save_period == 0 ||
            idx_episode == BitSim::Closer::n_episodes - 1) {
            save_params(idx_episode);
            interim_test();
        }
    }
}

void RL_Closer::save_params(int idx_period)
{

}

void RL_Closer::update_model(void)
{

}

void RL_Closer::interim_test(void)
{

}

RL_Action RL_Closer::get_action(RL_State state)
{

    if (step_total > BitSim::Closer::initial_random_action) {
        return environment.random_action();
    }

    return actor.get_action(state);
}

std::tuple<RL_State, bool> RL_Closer::step(RL_Action action)
{
    auto done = true;
    auto state = RL_State{};
    
    return std::make_tuple(state, done);
}
