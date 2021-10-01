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

    const auto command = std::string{ "make_simulator_data" };

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

        // Save deltas
        auto deltas = std::map<std::string, double>{};
        for (const auto symbol : BitBot::symbols) {
            const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
            deltas[symbol] = intrinsic_events->get_delta();
        }
        const auto deltas_file_path = std::string{ BitBot::path } + "/deltas.csv";
        auto deltas_file = std::ofstream{ deltas_file_path, std::ios::binary };
        deltas_file << "symbol,delta\n";
        for (const auto symbol : BitBot::symbols) {
            deltas_file << symbol << "," << deltas[symbol] << "\n";
        }
        deltas_file.close();
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
            const auto indicators = std::make_shared<Indicators>(symbol);
            const auto timestamp_start = time_point_ms{ date::sys_days(date::year{2020} / 7 / 1) };
            const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2021} / 9 / 25) };
            const auto path = std::string{ BitBot::path } + "/simulation_data";
            training_data.make(path, symbol, binance_klines, indicators, timestamp_start, timestamp_end);
        }
    }
    else if (command == "make_training_data_sections") {
        auto training_data = TrainingData{ };
        for (const auto symbol : BitBot::symbols) {
            printf("Make training data sections: %s\n", symbol);
            const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
            const auto indicators = std::make_shared<Indicators>(symbol);
            const auto timestamp_start = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) };
            const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2021} / 9 / 25) };
            const auto path = std::string{ BitBot::path } + "/training_data_sections";
            training_data.make(path, symbol, binance_klines, indicators, timestamp_start, timestamp_end);
            training_data.make_sections(path, symbol, binance_klines, indicators);
        }
    }
}
