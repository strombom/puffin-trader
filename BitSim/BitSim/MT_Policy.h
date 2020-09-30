#pragma once
#include "pch.h"

#include "BitLib/AggTicks.h"


class MT_Policy
{
public:
    MT_Policy(void);

    std::tuple<double, double, double> get_action(sptrAggTick agg_tick, double leverage);

private:

};

using sptrMT_Policy = std::shared_ptr<MT_Policy>;
