#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(double reward, double cart_position, double cart_velocity, double pole_angle, double pole_velocity);
    
    void set_done(void);
    bool is_done(void) const;
    torch::Tensor to_tensor(void) const;

    bool done;
    double cart_position;
    double cart_velocity;
    double pole_angle;
    double pole_velocity;
    double reward;
};

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
