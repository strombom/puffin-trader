#pragma once
#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "IntrinsicEvents.h"


class Indicators
{
public:
    Indicators(std::string symbol);

    void calculate(const sptrIntrinsicEvents intrinsic_events);

    void load(void);
    void save(void) const;

    std::string symbol;
    std::unique_ptr<std::array<std::array<float, BitBot::Indicators::indicator_width>, BitBot::Indicators::n_timestamps>> indicators;
    int n_steps;
};
