#include "pch.h"
#include "MT_Policy.h"


MT_Volatility::MT_Volatility(void) :
    initialized(false),
    buffer_pos(0)
{

}

void MT_Volatility::update(double price_low, double price_high)
{
    update_last_value(price_low, price_high);
    buffer_pos = (buffer_pos + 1) % buffer_low.size();
}

void MT_Volatility::update_last_value(double price_low, double price_high)
{
    if (!initialized) {
        buffer_low.fill(price_low);
        buffer_high.fill(price_high);
        initialized = true;
    }
    buffer_low[buffer_pos] = price_low;
    buffer_high[buffer_pos] = price_high;
}

double MT_Volatility::get(void) const
{
    const auto price_min = *std::min_element(std::begin(buffer_low), std::end(buffer_low));
    const auto price_max = *std::max_element(std::begin(buffer_high), std::end(buffer_high));
    const auto volatility = price_max / price_min - 1;

    return volatility;
}

MT_Policy::MT_Policy(void)
{

}

std::tuple<bool, double, double, double> MT_Policy::get_action(sptrAggTick agg_tick, double position_leverage)
{
    const auto pd_event = pd_events.update(agg_tick);

    volatility.update_last_value(agg_tick->low, agg_tick->high);
    const auto action_leverage = volatility.get() * 100;

    if (pd_event != nullptr) {
        volatility.update(agg_tick->low, agg_tick->high);

        const auto mark_price = (agg_tick->low + agg_tick->high) / 2;
        const auto direction = position_leverage >= 0 ? 1 : -1;
        const auto stop_loss_price = mark_price * (1 - direction * BitSim::Trader::Mech::stop_loss);
        const auto take_profit_price = mark_price * (1 + direction * BitSim::Trader::Mech::take_profit);

        return std::make_tuple(true, action_leverage, stop_loss_price, take_profit_price);
    }

    return std::make_tuple(false, action_leverage, 0, 0);
}
