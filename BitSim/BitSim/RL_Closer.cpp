#include "pch.h"

#include "RL_Closer.h"
#include "BitBotConstants.h"


void RL_Closer::train(void)
{


    for (auto idx_episode = 0; idx_episode < BitSim::Closer::n_episodes; ++idx_episode) {
        auto state = environment.reset();
        auto done = false;
        auto action = get_action(state);

        while (!done) {

            done = false;
        }
    }
}

RL_Closer::Action RL_Closer::get_action(State state)
{

    if (n_step_total > BitSim::Closer::initial_random_action) {
        return environment.random_action();
    }

    return actor.get_action(state);
}

RL_Closer::State RL_Closer::Environment::reset(void)
{
    return RL_Closer::State{};
}

RL_Closer::Action RL_Closer::Environment::random_action(void)
{
    return RL_Closer::Action{};
}

RL_Closer::Action RL_Closer::Actor::get_action(State state)
{
    return RL_Closer::Action{};
}
