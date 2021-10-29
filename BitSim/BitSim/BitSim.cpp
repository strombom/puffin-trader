#include "pch.h"
#include "Klines.h"
#include "Portfolio.h"
#include "Predictions.h"
#include "Simulator.h"
#include "Symbols.h"
#include "BitLib/BitBotConstants.h"

#include <algorithm>
#include <iostream>


int main()
{
    constexpr auto delta_idx = 5;
    constexpr auto threshold = 0.5;

    auto klines = Klines{};
    auto predictions = Predictions{};

    auto timestamp = klines.get_timestamp_start();
    const auto timestamp_end = klines.get_timestamp_end();

    auto portfolio = Portfolio{};

    while (timestamp < timestamp_end) {

        if (timestamp > date::sys_days(date::year{ 2020 } / 5 / 12) + std::chrono::hours{ 0 }) {
            auto a = 0;
        }

        klines.step_idx(timestamp);
        portfolio.set_mark_prices(klines);

        // Sell
        //portfolio.evaluate_positions(timestamp);

        // Buy
        predictions.step_idx(timestamp);
        for (const auto& symbol : symbols) {
            if (!predictions.has_prediction(symbol)) {
                continue;
            }
            
            const auto score = predictions.get_prediction_score(symbol, delta_idx);
            if (score > threshold) {
                if (!portfolio.has_available_position(symbol)) {
                    continue;
                }

                if (!portfolio.has_available_order(symbol)) {
                    portfolio.cancel_oldest_order(timestamp, symbol);
                }

                const auto equity = portfolio.get_equity();
                const auto cash = portfolio.get_cash();
                const auto position_value = min(equity / BitSim::Portfolio::total_capacity, cash * 0.99);
                const auto mark_price = klines.get_open_price(symbol);
                const auto position_size = position_value / mark_price * 0.99;

                if (position_size * mark_price > BitSim::min_position_value) {
                    //printf("%s Place limit order, %s %f %f\n", date::format("%F %T", timestamp).c_str(), symbol.name.data(), mark_price, position_size);
                    portfolio.place_limit_order(timestamp, symbol, delta_idx, position_size);
                }
            }
        }

        portfolio.evaluate_orders(timestamp, klines);

        timestamp += std::chrono::minutes{ 1 };
    }
}
