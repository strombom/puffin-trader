#include "pch.h"
#include "MT_Policy.h"


MT_Policy::MT_Policy(void)
{

}

std::tuple<double, double, double> MT_Policy::get_action(sptrAggTick agg_tick, double leverage)
{
    return std::make_tuple(0.0, 0.0, 0.0);
}
