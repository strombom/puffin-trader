#include "pch.h"
#include "Klines.h"
#include "Portfolio.h"
#include "Predictions.h"
#include "Simulator.h"
#include "Symbols.h"
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
    auto portfolio = Portfolio{};

    while (timestamp < timestamp_end) {
        klines.step_idx(timestamp);
        simulator.set_mark_price(klines);

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
                    portfolio.cancel_oldest_order(symbol);
                    simulator.cancel_orders();
                    portfolio.evaluate_orders();
                }

                const auto equity = simulator.get_equity();
                const auto cash = simulator.get_cash();
                const auto position_value = std::min(equity / BitSim::Portfolio::total_capacity, cash * 0.99);
                const auto mark_price = klines.get_open_price(symbol);
                const auto position_size = position_value / mark_price * 0.99;

                printf("%s Add limit order\n", date::format("%F %T", timestamp).c_str());
                auto order = simulator.limit_order(timestamp, symbol, position_size);
                portfolio.add_order(order);
            }
        }

        simulator.evaluate_orders(timestamp, klines);
        portfolio.evaluate_orders();
        //if (executed_orders->size() > 0) {
        //    portfolio.add_positions(std::move(executed_orders));
        //}

        timestamp += std::chrono::minutes{ 1 };
    }
}
