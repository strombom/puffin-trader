#include "pch.h"
#include "Indicators.h"
#include "PolyFit.h"


Indicators::Indicators(std::string symbol) : symbol(symbol) 
{
    indicators = std::make_unique<std::array<std::array<float, BitBot::Indicators::indicator_width>, BitBot::Indicators::n_timestamps>>();
}

void Indicators::calculate(const sptrIntrinsicEvents intrinsic_events)
{
    auto v = std::array<double, BitBot::Indicators::max_length>{};

    for (auto degree_idx = 0; degree_idx < BitBot::Indicators::degrees.size(); degree_idx++) {
        const auto degree = BitBot::Indicators::degrees[degree_idx];

        for (auto length_idx = 0; length_idx < BitBot::Indicators::lengths.size(); length_idx++) {
            const auto length = BitBot::Indicators::lengths[length_idx];
            auto poly_fit = PolyFit{degree, length};

            for (auto idx = length - 1; idx < intrinsic_events->events.size(); idx++) {

                for (auto i = 0; i < length; i++) {
                    v[i] = intrinsic_events->events[idx - length + i + 1].price;
                }

                poly_fit.calculate_direction(v);
            }
        }
    }
}
