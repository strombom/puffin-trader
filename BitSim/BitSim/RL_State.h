#pragma once
#include "pch.h"

class RL_State
{
public:
    RL_State(
        double reward,
        double cart_x_pos,
        double cart_x_vel,
        double cart_y_pos,
        double cart_y_vel,
        double pole_ang,
        double pole_vel
    );
    //RL_State(double reward, double angle, double velocity);
    RL_State(std::shared_ptr<RL_State> state);

    void set_done(void);
    bool is_done(void) const;
    torch::Tensor to_tensor(void) const;

    bool done;
    //double angle;
    //double velocity;
    double pole_ang;
    double pole_vel;
    double cart_x_pos;
    double cart_x_vel;
    double cart_y_pos;
    double cart_y_vel;
    double reward;
};

using sptrRL_State = std::shared_ptr<RL_State>;
