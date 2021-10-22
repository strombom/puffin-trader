#include "pch.h"
#include "Klines.h"
#include "Predictions.h"
#include "Simulator.h"
#include "BitLib/BitBotConstants.h"

#include <iostream>


int main()
{
    constexpr auto delta_idx = 5;
    constexpr auto threshold = 0.5;

    auto klines = Klines{};
    auto predictions = Predictions{};

    auto timestamp = klines.get_timestamp_start();
    const auto timestamp_end = klines.get_timestamp_end();

    auto simulator = Simulator{};

    while (timestamp < timestamp_end) {
        klines.step_idx(timestamp);
        simulator.set_mark_price(klines);

        predictions.step_idx(timestamp);

        for (const auto& symbol : BitBot::symbols) {
            if (!predictions.has_prediction(symbol)) {
                continue;
            }

            const auto score = predictions.get_prediction_score(symbol, delta_idx);
            if (score > threshold) {
                const auto equity = simulator.get_equity();
                const auto cash = simulator.get_cash();
            }
        }

        timestamp += std::chrono::minutes{ 1 };
    }
}
