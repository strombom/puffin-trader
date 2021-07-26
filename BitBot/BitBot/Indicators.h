#pragma once
#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "IntrinsicEvents.h"


class Indicators
{
public:
    Indicators(void) {}
    Indicators(std::string symbol);

    void calculate_and_save(std::string symbol, sptrIntrinsicEvents intrinsic_events);
    void join(void);

    void load(std::string symbol);

    std::vector<std::array<float, BitBot::Indicators::indicator_width>> indicators;
    std::vector<time_point_ms> timestamps;

private:
    std::vector<std::thread> threads;
};

using sptrIndicators = std::shared_ptr<Indicators>;
