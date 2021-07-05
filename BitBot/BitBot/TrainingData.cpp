#include "pch.h"
#include "TrainingData.h"

#include "BitLib/Logger.h"


void TrainingData::make(const sptrBinanceKlines binance_klines, const sptrIndicators indicators)
{
    make_ground_truth(binance_klines);
}

void TrainingData::make_ground_truth(const sptrBinanceKlines binance_klines)
{

    logger.info("make");
}