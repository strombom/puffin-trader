#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"


class TrainingData
{
public:
    TrainingData(const std::string& symbol);

    void make(const sptrBinanceKlines binance_klines, const sptrIndicators indicators);

private:
    void make_ground_truth(const sptrIndicators indicators);

    const std::string symbol;
    std::vector<int> ground_truth;
};

