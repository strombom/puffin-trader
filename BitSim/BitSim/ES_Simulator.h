#pragma once
#include "pch.h"


struct ES_State
{
public:
    double value_btc;
};


class ES_Simulator
{
public:
    ES_Simulator(void);

    virtual void reset(void) = 0;

    virtual ES_State market_order(double price, double volume) = 0;

private:

};

using sptrES_Simulator = std::shared_ptr<ES_Simulator>;
