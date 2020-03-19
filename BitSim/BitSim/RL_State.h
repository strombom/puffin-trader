#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(void) : done(false) {}
    
    void set_done(void);
    bool is_done(void);
    torch::Tensor to_tensor(void);

private:
    bool done;
};
