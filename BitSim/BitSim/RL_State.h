#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(double reward, torch::Tensor features, double leverage);
    
    void set_done(void);
    void set_reward(bool _reward);
    void set_state(torch::Tensor state, double leverage);

    bool is_done(void) const;
    double get_reward(void) const;
    torch::Tensor to_tensor(void) const;

private:
    bool done;
    double reward;
    torch::Tensor state;
};
