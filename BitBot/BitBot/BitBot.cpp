#include "pch.h"

#include "BinanceDownloadKlines.h"
#include "IntrinsicEvents.h"
#include "Indicators.h"
#include "TrainingData.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Logger.h"

#include <iostream>


int main()
{
    logger.info("BitSim started");

    const auto command = std::string{ "make_training_data" };

    if (command == "download_klines") {
        auto binance_download_klines = BinanceDownloadKlines{};
        binance_download_klines.download();
    }
    else if (command == "make_intrinsic_events") {
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);

            auto intrinsic_events = IntrinsicEvents{ symbol };
            intrinsic_events.insert(binance_klines);
            intrinsic_events.save();

            logger.info("Inserted %d events from %s", intrinsic_events.events.size(), symbol);
        }
    }
    else if (command == "make_indicators") {
        for (const auto symbol : BitBot::symbols) {
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);

            auto indicators = Indicators{ symbol };
            indicators.calculate(intrinsic_events);
            indicators.save();
        }
    }
    else if (command == "make_training_data") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            training_data.make(symbol, binance_klines, intrinsic_events, indicators);
            break;
        }
    }
}
