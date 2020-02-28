#pragma once
#include "pch.h"

class RL_Action
{
public:
    RL_Action(double buy_position, double buy_size, double sell_position, double sell_size) :
        buy_position(buy_position), buy_size(buy_size), sell_position(sell_position), sell_size(sell_size) {}
    
    static RL_Action random(void);

    double buy_position;
    double buy_size;
    double sell_position;
    double sell_size;

private:
};
