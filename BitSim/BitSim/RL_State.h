#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(double reward, double angle, double velocity);
    
    void set_done(void);
    bool is_done(void) const;
    torch::Tensor to_tensor(void) const;

    bool done;
    double angle;
    double velocity;
    double reward;
};

using sptrRL_State = std::shared_ptr<RL_State>;

/*
class RL_State
{
public:
    RL_State(double reward, torch::Tensor features, double leverage);

    void set_done(void);
    void set_state(torch::Tensor state, double leverage);

    bool is_done(void) const;
    double get_reward(void) const;
    torch::Tensor to_tensor(void) const;

private:
    bool done;
    double reward;
    torch::Tensor state;
};
*/
