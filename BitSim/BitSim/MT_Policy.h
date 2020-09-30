#pragma once
#include "pch.h"

#include "BitLib/AggTicks.h"


class MT_Volatility
{
public:
    MT_Volatility(void);

    double get_volatility(double price);

private:
    std::array<double, 250> buffer;
    bool initialized;
    int buffer_pos;
};


class MT_Policy
{
public:
    MT_Policy(void);

    std::tuple<double, double, double> get_action(sptrAggTick agg_tick, double leverage);

private:
    MT_Volatility volatility;

};

using sptrMT_Policy = std::shared_ptr<MT_Policy>;
