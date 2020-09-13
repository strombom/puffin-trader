#pragma once
#include "pch.h"

enum RL_Action_Direction
{
    dir_long,
    dir_short
};

class RL_Action
{
public:
    RL_Action(void) :
        direction(RL_Action_Direction::dir_long),
        stop_loss(0.0)
    {}

    RL_Action(torch::Tensor disc_action, torch::Tensor cont_action) :
        direction(disc_action[0].item().to<int>() == 0 ? RL_Action_Direction::dir_long : RL_Action_Direction::dir_short),
        stop_loss(cont_action[0].item().to<double>())
    {}

    //RL_Action(torch::Tensor cont_action, torch::Tensor disc_action) :
        //leverage(cont_action[0].item().to<double>()),
        //idle(disc_action[0].item().toLong() == 0),
        //limit_order(disc_action[0].item().toLong() == 1),
        //market_order(disc_action[0].item().toLong() == 2) {}

    RL_Action(RL_Action_Direction direction, double stop_loss) :
        direction(direction),
        stop_loss(stop_loss)
    {}

    //RL_Action(double leverage, bool idle, bool limit_order, bool market_order) :
    //    leverage(leverage),
    //    idle(idle),
    //    limit_order(limit_order),
    //    market_order(market_order) {}

    static std::shared_ptr<RL_Action> random(void);
    torch::Tensor to_tensor_cont(void) const;
    torch::Tensor to_tensor_disc(void) const;

    RL_Action_Direction direction;
    double stop_loss;
    //double min_profit;

    //double leverage;
    //bool idle;
    //bool limit_order;
    //bool market_order;

private:
};

using sptrRL_Action = std::shared_ptr<RL_Action>;

/*
// Cartpole

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
*/
