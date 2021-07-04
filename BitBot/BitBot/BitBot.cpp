#include "pch.h"

#include "BinanceDownloadKlines.h"
#include "IntrinsicEvents.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Logger.h"

#include <iostream>


int main()
{
    logger.info("BitSim started");

    const auto command = std::string{ "make_intrinsic_events" };

    if (command == "download_klines") {
        auto binance_download_klines = BinanceDownloadKlines{};
        binance_download_klines.download();
    }
    else if (command == "make_intrinsic_events") {
        for (const auto symbol : BitBot::symbols) {
            const auto file_path = std::string{ BitBot::Binance::Klines::path } + "\\" + symbol + ".dat";
            const auto binance_klines = std::make_shared<BinanceKlines>(file_path);
            auto intrinsic_events = IntrinsicEvents{ symbol };
            intrinsic_events.insert(binance_klines);
            intrinsic_events.save();

            logger.info("Inserted %d events from %s", intrinsic_events.events.size(), symbol);
        }
    }
}
