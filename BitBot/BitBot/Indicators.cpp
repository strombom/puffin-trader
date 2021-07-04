#include "pch.h"
#include "Indicators.h"


Indicators::Indicators(std::string symbol) : symbol(symbol) 
{
    indicators = std::make_unique<std::array<std::array<float, BitBot::Indicators::indicator_width>, BitBot::Indicators::n_timestamps>>();
}

void Indicators::calculate(const sptrIntrinsicEvents intrinsic_events)
{
    for (auto degree_idx = 0; degree_idx < BitBot::Indicators::degrees.size(); degree_idx++) {
        const auto degree = BitBot::Indicators::degrees[degree_idx];

        for (auto length_idx = 0; length_idx < BitBot::Indicators::lengths.size(); length_idx++) {
            const auto length = BitBot::Indicators::lengths[length_idx];
            
            for (auto idx = length - 1; idx < indicators->size(); idx++) {

            }
        }
    }
}
