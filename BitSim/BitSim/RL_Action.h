#pragma once
#include "pch.h"

class RL_Action
{
public:
    RL_Action(torch::Tensor action) :
        buy_position(action[0].item().to<double>()), 
        buy_size(action[1].item().to<double>()), 
        sell_position(action[2].item().to<double>()), 
        sell_size(action[3].item().to<double>()) {}

    RL_Action(double buy_position, double buy_size, double sell_position, double sell_size) :
        buy_position(buy_position), 
        buy_size(buy_size), 
        sell_position(sell_position), 
        sell_size(sell_size) {}
    
    static RL_Action random(void);

    double buy_position;
    double buy_size;
    double sell_position;
    double sell_size;

private:
};
