#include "pch.h"

#include "TrainingData.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"


TrainingData::TrainingData(const std::string& symbol) : symbol(symbol)
{

}

void TrainingData::make(const sptrBinanceKlines binance_klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators)
{
    make_ground_truth(intrinsic_events);
}

void TrainingData::make_ground_truth(const sptrIntrinsicEvents intrinsic_events)
{
    ground_truth.resize(intrinsic_events->events.size());

    auto positions = std::list<Position>{};

    for (auto idx = 0; idx < intrinsic_events->events.size(); idx++) {
        const auto mark_price = intrinsic_events->events.at(idx).price;

        auto remove = false;
        for (auto&& position : positions) {
            if (mark_price >= position.take_profit) {
                ground_truth.at(position.idx) = 1;
                position.remove = true;
                remove = true;
            }
            else if (mark_price <= position.stop_loss) {
                ground_truth.at(position.idx) = -1;
                position.remove = true;
                remove = true;
            }
        }

        if (remove) {
            positions.remove_if([](const Position& position) { return position.remove; });
        }

        positions.emplace_back(
            idx, 
            mark_price * BitBot::TrainingData::take_profit, 
            mark_price * BitBot::TrainingData::stop_loss
        );
    }
}