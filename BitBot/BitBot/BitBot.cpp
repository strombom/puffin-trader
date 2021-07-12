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

    const auto command = std::string{ "make_training_data_sections" };

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
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            training_data.make(symbol, intrinsic_events, indicators);
        }
    }
    else if (command == "make_training_data_sections") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            auto year = 2020;
            auto month = 1;
            while (!(year == 2020 && month == 12)) {
                const auto timestamp_start = time_point_ms{ date::sys_days(date::year{year} / month / 1) + std::chrono::hours{ 0 } };
                const auto timestamp_end = timestamp_start + date::months{ 7 };
                training_data.make_section(symbol, intrinsic_events, indicators, timestamp_start, timestamp_end);

                month++;
                if (month == 13) {
                    month = 1;
                    year++;
                }
            }

        }
    }
}
