#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"



class TrainingData
{
public:
    TrainingData();

    void make(const std::string& symbol, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end);
    void make_section(const std::string& symbol, const std::string& suffix, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end);

    void join(void);

private:
    void make_ground_truth(const std::string symbol, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events);

    struct Position;
    std::shared_ptr<std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>> ground_truth;
    std::shared_ptr<std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>> ground_truth_timestamps;
    std::string ground_truth_symbol;

    std::vector<std::thread> threads;
};

struct TrainingData::Position
{
public:
    Position(int ie_idx, float take_profit, float stop_loss) :
        ie_idx(ie_idx), take_profit(take_profit), stop_loss(stop_loss), remove(false) {}

    int ie_idx;
    float take_profit;
    float stop_loss;
    bool remove;
};
