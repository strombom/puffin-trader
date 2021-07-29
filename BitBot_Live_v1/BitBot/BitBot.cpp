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
    logger.info("BitBot Live started");

    while (true) {
        auto binance_download_klines = BinanceDownloadKlines{};
        binance_download_klines.download();


    }

    const auto command = std::string{ "make_training_data_sections" };

    if (command == "download_klines") {
        auto binance_download_klines = BinanceDownloadKlines{};
        binance_download_klines.download();
    }
    else if (command == "make_intrinsic_events") {
        auto intrinsic_events = IntrinsicEvents{};
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);

            intrinsic_events.calculate_and_save(symbol, binance_klines);
        }
        intrinsic_events.join();
    }
    else if (command == "make_indicators") {
        auto indicators = Indicators{};
        for (const auto symbol : BitBot::symbols) {
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);

            indicators.calculate_and_save(symbol, intrinsic_events);
        }
        indicators.join();
    }
    else if (command == "make_simulator_data") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);

            const auto timestamp_start = time_point_ms{ date::sys_days(date::year{2021} / 01 / 01) };
            const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2021} / 07 / 22) };
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
            while (day < 208) {
                const auto timestamp_start = time_point_ms{ date::sys_days(date::year{year} / 01 / 01) + date::days{day}};
                const auto timestamp_end = timestamp_start + date::months{ 12 };
                printf("%s %s, %d\n", symbol, DateTime::to_string_iso_8601(timestamp_start).c_str(), day);
                training_data.make_section(symbol, "train", binance_klines, intrinsic_events, indicators, timestamp_start, timestamp_end);
                training_data.make_section(symbol, "valid", binance_klines, intrinsic_events, indicators, timestamp_end, timestamp_end + date::days{ 1 });

                day += 2;
            }
            //break;
        }

        training_data.join();
    }
}
