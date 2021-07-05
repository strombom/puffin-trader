#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"


class TrainingData
{
public:
    TrainingData(const std::string &symbol) : symbol(symbol) {}

    void make(const sptrBinanceKlines binance_klines, const sptrIndicators indicators);

private:
    void make_ground_truth(const sptrBinanceKlines binance_klines);

    const std::string symbol;
    std::array<int, BitBot::n_timestamps> ground_truth;
};

