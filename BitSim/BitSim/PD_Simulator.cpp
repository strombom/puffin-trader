#include "pch.h"

#include "BitLib/Utils.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks) :
    agg_ticks(agg_ticks)
{
    exchange = std::make_shared<ES_Bitmex>();
    events = std::make_shared<PD_Events>(agg_ticks);

    training_start = agg_ticks->rows.front().timestamp;
    training_end = agg_ticks->rows.front().timestamp + (agg_ticks->rows.back().timestamp - agg_ticks->rows.front().timestamp) * 4 / 5;
    validation_start = training_end;
    validation_end = agg_ticks->rows.back().timestamp;
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    auto timestamp_start = agg_ticks->rows.front().timestamp;
    if (validation) {
        timestamp_start = Utils::random(validation_start, validation_end - BitSim::Trader::episode_length);
    }
    else {
        timestamp_start = Utils::random(training_start, training_end - BitSim::Trader::episode_length);
    }
    episode_end = timestamp_start + BitSim::Trader::episode_length;

    // Find start index
    agg_ticks_idx = 0;
    while (agg_ticks_idx < agg_ticks->rows.size() && agg_ticks->rows[agg_ticks_idx].timestamp < timestamp_start)
    {
        agg_ticks_idx++;
    }

    const auto price = agg_ticks->rows[agg_ticks_idx].low;
    orderbook_last_price = price;
    exchange->reset(price);
    events->rewind();

    time_since_leverage_change = 0;

    const auto reward = 0.0;
    const auto features = make_features();
    const auto leverage = 0.0;
    const auto delta_price = 0.0;
    const auto time_since_change = 0.0;

    return std::make_shared<RL_State>(reward, features, leverage, delta_price, time_since_change);
}

sptrRL_State PD_Simulator::step(sptrRL_Action action)
{
    const auto previous_event = events->get_event();
    const auto previous_agg_tick = &agg_ticks->rows[previous_event->agg_tick_idx];

    // Stop loss?

    if (previous_agg_tick->high > orderbook_last_price + 0.5) {
        orderbook_last_price = previous_agg_tick->high - 0.5;
    }
    else if (previous_agg_tick->low < orderbook_last_price) {
        orderbook_last_price = previous_agg_tick->low;
    }
    const auto mark_price = orderbook_last_price;

    const auto order_leverage = action->direction == RL_Action_Direction::dir_long ? BitSim::BitMex::max_leverage : -BitSim::BitMex::max_leverage;
    const auto position_leverage = exchange->get_leverage(mark_price);

    if (order_leverage > 0 && position_leverage < 0 ||
        order_leverage < 0 && position_leverage > 0) {
        time_since_leverage_change = 0;
        const auto order_contracts = exchange->calculate_order_size(order_leverage, mark_price);
        exchange->market_order(order_contracts, mark_price);
    }
    else {
        time_since_leverage_change += 1;
    }
    const auto time_since_change = std::log1p(time_since_leverage_change) / 5.0;
    auto delta_price = exchange->get_position_price();
    delta_price = (delta_price < 0 ? -1.0 : 1.0) * std::log1p(std::abs(delta_price)) / 3.0;

    const auto reward = calculate_reward();
    const auto features = make_features();
    const auto leverage = exchange->get_leverage(mark_price);

    auto state = std::make_shared<RL_State>(reward, features, leverage, delta_price, time_since_change);

    //++agg_ticks_idx;
    //if (agg_ticks->rows[agg_ticks_idx].timestamp >= episode_end) {
    //    state->set_done();
    //}

    if (!events->step()) {
        state->set_done();
    }

    return state;
}

time_point_ms PD_Simulator::get_start_timestamp(void)
{
    return system_clock_ms_now(); // simulator->get_start_timestamp();
}

torch::Tensor PD_Simulator::make_features(void)
{
    auto features = torch::Tensor{};
    return features;
}

double PD_Simulator::calculate_reward(void)
{
    return 0.0;
}
