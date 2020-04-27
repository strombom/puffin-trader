#pragma once
#include "pch.h"

class RL_State
{
public:
    //RL_State(double reward, double cart_position, double cart_velocity, double pole_angle, double pole_velocity);
    RL_State(double reward, double angle, double velocity);
    
    void set_done(void);
    bool is_done(void) const;
    torch::Tensor to_tensor(void) const;

    bool done;
    double angle;
    double velocity;
    //double pole_angle;
    //double pole_velocity;
    //double cart_position;
    //double cart_velocity;
    double reward;
};

using sptrRL_State = std::shared_ptr<RL_State>;
