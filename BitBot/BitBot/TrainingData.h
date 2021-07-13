#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"



class TrainingData
{
public:
    TrainingData();

    void make(const std::string& symbol, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators);
    void make_section(const std::string& symbol, const std::string& suffix, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end);

private:
    void make_ground_truth(const std::string symbol, const sptrIntrinsicEvents intrinsic_events);

    struct Position;
    std::vector<int> ground_truth;
    std::vector<time_point_ms> ground_truth_timestamps;
    std::string ground_truth_symbol;
};

struct TrainingData::Position
{
public:
    Position(int idx, float take_profit, float stop_loss) :
        idx(idx), take_profit(take_profit), stop_loss(stop_loss), remove(false) {}

    int idx;
    float take_profit;
    float stop_loss;
    bool remove;
};
