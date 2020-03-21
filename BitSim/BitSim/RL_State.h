#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(void) :
        done(false),
        reward(0.0) {}

    RL_State(double reward) : 
        done(false),
        reward(reward) {}
    
    void set_done(void);
    bool is_done(void);
    torch::Tensor to_tensor(void);

private:
    bool done;
    double reward;
};
