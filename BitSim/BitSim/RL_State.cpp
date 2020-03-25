#include "pch.h"
#include "RL_State.h"


void RL_State::set_done(void)
{
    done = true;
}

bool RL_State::is_done(void)
{
    return done;
}

torch::Tensor RL_State::to_tensor(void) const
{
    return torch::tensor({ reward });
}
