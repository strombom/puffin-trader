#pragma once
#include "pch.h"


class RL_Action
{
public:
    RL_Action(void) :
        move(0.0) {}

    RL_Action(torch::Tensor action) :
        move(action[0].item().to<double>()) {}

    RL_Action(double move) :
        move(move) {}

    static RL_Action random(void);
    torch::Tensor to_tensor(void) const;

    double move;

private:
};

/*
class RL_Action
{
public:
    RL_Action(void) :
        buy_action(0.0),
        buy_size(0.0),
        sell_action(0.0),
        sell_size(0.0) {}

    RL_Action(torch::Tensor action) :
        buy_action(action[0].item().to<double>()), 
        buy_size(action[1].item().to<double>()), 
        sell_action(action[2].item().to<double>()), 
        sell_size(action[3].item().to<double>()) {}

    RL_Action(double buy_action, double buy_size, double sell_action, double sell_size) :
        buy_action(buy_action),
        buy_size(buy_size), 
        sell_action(sell_action),
        sell_size(sell_size) {}
    
    static RL_Action random(void);
    torch::Tensor to_tensor(void) const;

    double buy_action;
    double buy_size;
    double sell_action;
    double sell_size;

private:
};
*/
