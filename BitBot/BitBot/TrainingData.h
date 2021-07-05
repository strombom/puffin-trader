#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"



class TrainingData
{
public:
    TrainingData();

    void make(const std::string& symbol, const sptrBinanceKlines binance_klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators);

private:
    std::vector<int> make_ground_truth(const sptrIntrinsicEvents intrinsic_events);

    struct Position;
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
