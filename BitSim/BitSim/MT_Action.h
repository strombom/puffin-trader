#pragma once
#include "pch.h"


class MT_Action
{
public:
    MT_Action(void) :
        buy(false),
        stop_loss(0.0),
        min_profit(0.0) {}

    MT_Action(bool buy, double stop_loss, double min_profit) :
        buy(buy),
        stop_loss(stop_loss),
        min_profit(min_profit) {}

private:
    bool buy;
    double stop_loss;
    double min_profit;
};

using sptrMT_Action = std::shared_ptr<MT_Action>;

