#pragma once
#include "pch.h"


class RL_Action
{
public:
    RL_Action(void) :
        x_move_not(true), x_move_left(false), x_move_right(false),
        y_move(0.0)
    {}

    RL_Action(torch::Tensor cont_action, torch::Tensor disc_action) :
        y_move(cont_action[0].item().to<double>()), 
        x_move_not(disc_action[0].item().toLong() == 0), //x_move_not(true), //
        x_move_left(disc_action[0].item().toLong() == 1),
        x_move_right(disc_action[0].item().toLong() == 2)
    {}

    RL_Action(bool x_move_not, bool x_move_left, bool x_move_right, double y_move) :
        x_move_not(x_move_not), x_move_left(x_move_left), x_move_right(x_move_right),
        y_move(y_move) {}

    static std::shared_ptr<RL_Action> random(void);
    torch::Tensor to_tensor_cont(void) const;
    torch::Tensor to_tensor_disc(void) const;

    //double move_side;
    bool x_move_not;
    bool x_move_left;
    bool x_move_right;
    double y_move;

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
