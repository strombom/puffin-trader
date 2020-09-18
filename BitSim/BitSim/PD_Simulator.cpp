#include "pch.h"

#include "BitLib/Utils.h"
#include "PD_Simulator.h"


PD_Simulator::PD_Simulator(sptrAggTicks agg_ticks) :
    agg_ticks(agg_ticks)
{
    exchange = std::make_shared<ES_Bitmex>();
    events = std::make_shared<PD_Events>(agg_ticks);

    training_start = agg_ticks->agg_ticks.front().timestamp;
    training_end = agg_ticks->agg_ticks.front().timestamp + (agg_ticks->agg_ticks.back().timestamp - agg_ticks->agg_ticks.front().timestamp) * 4 / 5;
    validation_start = training_end;
    validation_end = agg_ticks->agg_ticks.back().timestamp;
}

sptrRL_State PD_Simulator::reset(int idx_episode, bool validation)
{
    auto timestamp_start = time_point_ms{};
    if (validation) {
        timestamp_start = Utils::random(validation_start, validation_end - BitSim::Trader::episode_length);
    }
    else {
        timestamp_start = Utils::random(training_start, training_end - BitSim::Trader::episode_length);
    }
    episode_end = timestamp_start + BitSim::Trader::episode_length;

    // Find start index
    pd_events_idx = 0;
    while (pd_events_idx < events->events.size() && events->events[pd_events_idx].timestamp < timestamp_start)
    {
        pd_events_idx++;
    }

    const auto price = agg_ticks->agg_ticks[events->events[pd_events_idx].agg_tick_idx].low;
    exchange->reset(price);
    //pd_events_idx = BitSim::Trader::feature_events_count;

    position_timestamp = events->events[pd_events_idx].timestamp;
    position_price = price;
    position_direction = 1;
    position_stop_loss = price * (1 - position_direction * BitSim::Trader::stop_loss_range);

    const auto reward = 0.0;
    const auto features = make_features(position_timestamp, position_price);
    const auto leverage = 0.0;
    const auto delta_price = 0.0;
    const auto time_since_change = 0.0;

    return std::make_shared<RL_State>(reward, features, leverage, delta_price, time_since_change);
}

sptrRL_State PD_Simulator::step(sptrRL_Action action)
{
    const auto event = &events->events[pd_events_idx];
    const auto agg_tick = &agg_ticks->agg_ticks[event->agg_tick_idx];

    const auto mark_price = (agg_tick->high  + agg_tick->low) / 2;
    const auto order_leverage = action->direction == RL_Action_Direction::dir_long ? BitSim::BitMex::max_leverage : -BitSim::BitMex::max_leverage;
    const auto position_leverage = exchange->get_leverage(mark_price);

    if (order_leverage > 0 && position_leverage <= 0 ||
        order_leverage < 0 && position_leverage >= 0) {
        const auto order_contracts = exchange->calculate_order_size(order_leverage, mark_price);
        exchange->market_order(order_contracts, mark_price);
        position_direction = order_contracts >= 0 ? 1 : -1;
        position_timestamp = agg_tick->timestamp;
        position_price = agg_tick->high;
        position_stop_loss = position_price * (1 - position_direction * BitSim::Trader::stop_loss_range);
    }

    const auto time_since_change = (double)(agg_tick->timestamp - position_timestamp).count(); // std::log1p((agg_tick->timestamp - position_timestamp).count()) / 5.0;
    auto delta_price = exchange->get_position_price();
    //delta_price = (delta_price < 0 ? -1.0 : 1.0) * std::log1p(std::abs(delta_price)) / 3.0;

    const auto reward = calculate_reward();
    const auto features = make_features(agg_tick->timestamp, mark_price);
    const auto leverage = exchange->get_leverage(mark_price);

    auto state = std::make_shared<RL_State>(reward, features, leverage, delta_price, time_since_change);

    ++pd_events_idx;
    if (pd_events_idx >= events->events.size() || events->events[pd_events_idx].timestamp >= episode_end) {
        state->set_done();
    }
    else {
        // Check stop loss
        const auto next_event = &events->events[pd_events_idx];
        const auto next_agg_tick = &agg_ticks->agg_ticks[next_event->agg_tick_idx];
        for (auto agg_tick_idx = event->agg_tick_idx; agg_tick_idx < next_event->agg_tick_idx; ++agg_tick_idx) {
            const auto agg_tick = &agg_ticks->agg_ticks[agg_tick_idx];
            if (position_direction == 1 && agg_tick->low < position_stop_loss) {
                position_direction = -1;
                position_timestamp = agg_tick->timestamp;
                position_price = agg_tick->low;
                const auto order_contracts = exchange->calculate_order_size(-BitSim::BitMex::max_leverage, position_price);
                exchange->market_order(order_contracts, position_price);
                break;
            }
            else if (position_direction == -1 && agg_tick->high > position_stop_loss) {
                position_direction = 1;
                position_timestamp = agg_tick->timestamp;
                position_price = agg_tick->high;
                const auto order_contracts = exchange->calculate_order_size(BitSim::BitMex::max_leverage, position_price);
                exchange->market_order(order_contracts, position_price);
                break;
            }
        }
    }

    return state;
}

time_point_ms PD_Simulator::get_start_timestamp(void)
{
    return system_clock_ms_now(); // simulator->get_start_timestamp();
}

torch::Tensor PD_Simulator::make_features(time_point_ms ref_timestamp, double ref_price)
{
    constexpr auto features_size = 3 + 2 * BitSim::Trader::feature_events_count;
    auto features = torch::empty({ 1, features_size });
    auto features_access = features.accessor<float, 2>();
    
    const auto ref_event = &events->events[pd_events_idx];
    const auto ref_agg_tick = &agg_ticks->agg_ticks[ref_event->agg_tick_idx];

    for (auto idx = 0; idx < BitSim::Trader::feature_events_count; idx++) {
        const auto event = &events->events[pd_events_idx - (idx + 1)];
        const auto agg_tick = &agg_ticks->agg_ticks[event->agg_tick_idx];

        auto delta_time = (float)(ref_timestamp - agg_tick->timestamp).count();
        auto delta_price = (float)(ref_price - (agg_tick->high + agg_tick->low) / 2);

        delta_price = delta_price / 64;
        delta_time = (float)std::log1p(delta_time) / 8 - 1.4f;

        features_access[0][static_cast<long long>(idx) * 2 + 0] = delta_time;
        features_access[0][static_cast<long long>(idx) * 2 + 1] = delta_price;
    }

    const auto leverage = exchange->get_leverage(ref_price);

    features_access[0][features_size - 3] = (float)std::log1p((ref_timestamp - position_timestamp).count()) / 8 - 1.4f; // Time since last order
    features_access[0][features_size - 2] = (float)(ref_price - position_price) / 64; // Price diff since last order
    features_access[0][features_size - 1] = (float)(leverage) / 10;

    return features;
}

double PD_Simulator::calculate_reward(void)
{
    return 0.0;
}
