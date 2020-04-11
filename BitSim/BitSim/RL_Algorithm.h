#pragma once

#include "RL_State.h"
#include "RL_Action.h"


class RL_Algorithm
{
public:
    virtual std::shared_ptr<RL_Action> get_action(sptrRL_State state) = 0;
    virtual std::shared_ptr<RL_Action> get_random_action(sptrRL_State state) = 0;
    virtual std::array<double, 6> update_model(void) = 0;
    virtual void append_to_replay_buffer(sptrRL_State current_state, sptrRL_Action action, sptrRL_State next_state, bool done) = 0;

    //virtual void save(const std::string& filename) = 0;
    //virtual void open(const std::string& filename) = 0;
};

using uptrRL_Algorithm = std::unique_ptr<RL_Algorithm>;
