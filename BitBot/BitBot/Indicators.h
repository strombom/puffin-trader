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

    const std::string symbol;
    std::vector<std::array<float, BitBot::Indicators::indicator_width>> indicators;
    //int n_steps;
};

using sptrIndicators = std::shared_ptr<Indicators>;
