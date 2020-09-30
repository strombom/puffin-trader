#include "pch.h"
#include "MT_Policy.h"


MT_Volatility::MT_Volatility(void) :
    initialized(false),
    buffer_pos(0)
{

}

double MT_Volatility::get_volatility(double price)
{
    if (!initialized) {
        buffer.fill(price);
        initialized = true;
    }
    buffer[buffer_pos] = price;
    buffer_pos = (buffer_pos + 1) % buffer.size();

    const auto price_max = *std::max_element(std::begin(buffer), std::end(buffer));
    const auto price_min = *std::max_element(std::begin(buffer), std::end(buffer));
    const auto volatility = price_max / price_min - 1;

    return 0.0;
}

MT_Policy::MT_Policy(void)
{

}

std::tuple<double, double, double> MT_Policy::get_action(sptrAggTick agg_tick, double leverage)
{
    return std::make_tuple(0.0, 0.0, 0.0);
}
