#pragma once
#include "pch.h"


class RL_Action
{
public:
    RL_Action(void) :
        idle(true),
        b9(false),
        b3(false),
        s1(false),
        s4(false),
        s10(false) {}
        //leverage(0.0),
        //idle(true),
        //limit_order(false),
        //market_order(false) {}

    RL_Action(torch::Tensor disc_action) :
        idle(disc_action[0].item().toLong() == 0),
        b9(disc_action[0].item().toLong() == 1),
        b3(disc_action[0].item().toLong() == 2),
        s1(disc_action[0].item().toLong() == 3),
        s4(disc_action[0].item().toLong() == 4),
        s10(disc_action[0].item().toLong() == 5) {}

    //RL_Action(torch::Tensor cont_action, torch::Tensor disc_action) :
        //leverage(cont_action[0].item().to<double>()),
        //idle(disc_action[0].item().toLong() == 0),
        //limit_order(disc_action[0].item().toLong() == 1),
        //market_order(disc_action[0].item().toLong() == 2) {}

    RL_Action(bool idle, bool b9, bool b3, bool s1, bool s4, bool s10) :
        idle(idle),
        b9(b9),
        b3(b3),
        s1(s1),
        s4(s4),
        s10(s10) {}

    //RL_Action(double leverage, bool idle, bool limit_order, bool market_order) :
    //    leverage(leverage),
    //    idle(idle),
    //    limit_order(limit_order),
    //    market_order(market_order) {}

    static std::shared_ptr<RL_Action> random(void);
    //torch::Tensor to_tensor_cont(void) const;
    torch::Tensor to_tensor_disc(void) const;

    bool idle;
    bool b9;
    bool b3;
    bool s1;
    bool s4;
    bool s10;
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
