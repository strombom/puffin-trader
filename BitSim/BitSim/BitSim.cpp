#include "pch.h"
#include "Klines.h"
#include "Predictions.h"
#include "Simulator.h"
#include "BitLib/BitBotConstants.h"

#include <iostream>


int main()
{
    auto klines = Klines{};
    auto predictions = Predictions{};

    auto timestamp = klines.get_timestamp_start();
    const auto timestamp_end = klines.get_timestamp_end();

    auto simulator = Simulator{};

    while (timestamp < timestamp_end) {
        klines.step_idx(timestamp);
        predictions.step_idx(timestamp);

        simulator.set_mark_price(klines);

        timestamp += std::chrono::minutes{ 1 };
    }
}
