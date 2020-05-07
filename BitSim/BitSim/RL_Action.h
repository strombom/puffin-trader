#pragma once
#include "pch.h"


class RL_Action
{
public:
    RL_Action(void) :
        //move_side(0.0),
        move_not(true), move_up(false), move_down(false)
    {}

    RL_Action(torch::Tensor cont_action)// :
        //move_side(action[0].item().to<double>())
    {}

    RL_Action(torch::Tensor cont_action, torch::Tensor disc_action) :
        //move_side(action[0].item().to<double>()),
        move_not(disc_action[0].item().toLong() == 0),
        move_up(disc_action[0].item().toLong() == 1),
        move_down(disc_action[0].item().toLong() == 2) {}

    RL_Action(bool move_not, bool move_up, bool move_down) :
        //move_side(move_side),
        move_not(move_not), move_up(move_up), move_down(move_down) {}

    static std::shared_ptr<RL_Action> random(void);
    torch::Tensor to_tensor_cont(void) const;
    torch::Tensor to_tensor_disc(void) const;

    //double move_side;
    bool move_not;
    bool move_up;
    bool move_down;

private:
};

using sptrRL_Action = std::shared_ptr<RL_Action>;

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
