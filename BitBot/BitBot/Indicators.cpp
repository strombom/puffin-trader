#include "pch.h"
#include "Indicators.h"
#include "PolyFit.h"
#include "BitLib/Logger.h"


Indicators::Indicators(std::string symbol) : symbol(symbol) 
{
    indicators = std::make_unique<std::array<std::array<float, BitBot::Indicators::indicator_width>, BitBot::Indicators::n_timestamps>>();
}

void Indicators::calculate(const sptrIntrinsicEvents intrinsic_events)
{
    auto price_steps = std::array<double, BitBot::Indicators::max_length>{};

    for (auto degree_idx = 0; degree_idx < BitBot::Indicators::degrees.size(); degree_idx++) {
        const auto degree = BitBot::Indicators::degrees[degree_idx];

        for (auto length_idx = 0; length_idx < BitBot::Indicators::lengths.size(); length_idx++) {
            const auto length = BitBot::Indicators::lengths[length_idx];

            const auto indicator_idx = degree_idx * BitBot::Indicators::lengths.size() * 2 + length_idx * 2;
            logger.info("degree_idx(%d) length_idx(%d) indicator_idx(%d)", degree_idx, length_idx, indicator_idx);

            auto poly_fit = PolyFit{degree, length};

            for (auto idx = max_length - 1; idx < intrinsic_events->events.size(); idx++) {
                for (auto i = 0; i < length; i++) {
                    price_steps[i] = intrinsic_events->events[idx - length + i + 1].price;
                }

                const auto [p0, p1] = poly_fit.calculate_direction(price_steps);
                const auto price = intrinsic_events->events[idx].price;
                const auto price_diff = p1 / price - 1.0;
                const auto direction = p1 / p0 - 1.0;

                indicators->at(idx).at(indicator_idx + 0) = (float)price_diff;
                indicators->at(idx).at(indicator_idx + 1) = (float)direction;
                //logger.info("%f, %f, %f, %f, %f", p0, p1, price, price_diff, direction);
            }
        }
    }
}
