#pragma once
#include "pch.h"

#include "PD_Events.h"
#include "BitLib/AggTicks.h"


class MT_Volatility
{
public:
    MT_Volatility(void);

    void update(double price_low, double price_high);
    void update_last_value(double price_low, double price_high);
    double get(void) const;

private:
    std::array<double, 250> buffer_high;
    std::array<double, 250> buffer_low;
    bool initialized;
    int buffer_pos;
};


class MT_Policy
{
public:
    MT_Policy(void);

    std::tuple<bool, double, double, double> get_action(sptrAggTick agg_tick, double position_leverage);

private:
    MT_Volatility volatility;
    PD_Events pd_events;

};

using sptrMT_Policy = std::shared_ptr<MT_Policy>;
