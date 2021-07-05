#include "pch.h"
#include "TrainingData.h"

#include "BitLib/Logger.h"


TrainingData::TrainingData(const std::string& symbol) : symbol(symbol)
{

}

void TrainingData::make(const sptrBinanceKlines binance_klines, const sptrIndicators indicators)
{
    make_ground_truth(indicators);
}

void TrainingData::make_ground_truth(const sptrIndicators indicators)
{
    ground_truth.clear();

    //for (auto idx = 0; idx < indicators->n_steps; idx++) {

    //}

    logger.info("make");
}