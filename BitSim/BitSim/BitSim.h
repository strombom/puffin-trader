#pragma once


namespace BitSim {
    constexpr auto n_batches = 100;
    constexpr auto batch_size = 20;
    constexpr auto observation_length = 160;
    constexpr auto n_channels = 3;
    constexpr auto n_observations = 5;
    constexpr auto n_predictions = 1;
    constexpr auto n_positive = 1;
    constexpr auto n_negative = 19;

    constexpr auto feature_size = 256;

    const auto ch_price = 0;
    const auto ch_buy_volume = 1;
    const auto ch_sell_volume = 2;
}
