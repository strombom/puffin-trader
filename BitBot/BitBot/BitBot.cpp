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
            intrinsic_events.calculate(binance_klines);
            intrinsic_events.save();

            logger.info("Inserted %d events from %s, delta: %f", intrinsic_events.events.size(), symbol, intrinsic_events.delta);
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
    else if (command == "make_simulator_data") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            const auto timestamp_start = time_point_ms{ date::sys_days(date::year{2021} / 01 / 01) };
            const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2021} / 07 / 19) };
            training_data.make(symbol, binance_klines, intrinsic_events, indicators, timestamp_start, timestamp_end);
        }
    }
    else if (command == "make_training_data_sections") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            auto year = 2020;
            auto day = 0;
            while (day < 204) {
                const auto timestamp_start = time_point_ms{ date::sys_days(date::year{year} / 01 / 01) + date::days{day}};
                const auto timestamp_end = timestamp_start + date::months{ 12 };
                training_data.make_section(symbol, "train", binance_klines, intrinsic_events, indicators, timestamp_start, timestamp_end);
                training_data.make_section(symbol, "valid", binance_klines, intrinsic_events, indicators, timestamp_end, timestamp_end + date::days{ 1 });

                day += 2;
            }
        }
    }
}
