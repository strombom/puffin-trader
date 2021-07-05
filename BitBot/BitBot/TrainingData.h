#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"


struct Position
{
public:
    Position(int idx, float take_profit, float stop_loss) :
        idx(idx), take_profit(take_profit), stop_loss(stop_loss), remove(false) {}

    int idx;
    float take_profit;
    float stop_loss;
    bool remove;
};


class TrainingData
{
public:
    TrainingData(const std::string& symbol);

    void make(const sptrBinanceKlines binance_klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators);

private:
    void make_ground_truth(const sptrIntrinsicEvents intrinsic_events);

    const std::string symbol;
    std::vector<int> ground_truth;
};

