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

    // Download klines
    auto binance_download_klines = BinanceDownloadKlines{};
    binance_download_klines.download();

    // Make intrinsic events
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
    const auto deltas_file_path = std::string{ BitBotLiveV1::path } + "/deltas.csv";
    auto deltas_file = std::ofstream{ deltas_file_path, std::ios::binary };
    deltas_file << "symbol,delta\n";
    for (const auto symbol : BitBot::symbols) {
        deltas_file << symbol << "," << deltas[symbol] << "\n";
    }
    deltas_file.close();

    // Make indicators
    auto indicators = Indicators{};
    for (const auto symbol : BitBot::symbols) {
        const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
        indicators.calculate_and_save(symbol, intrinsic_events);
    }
    indicators.join();

    // Make training data
    auto training_data = TrainingData{ };
    for (const auto symbol : BitBot::symbols) {
        const auto binance_klines = std::make_shared<BinanceKlines>(symbol);
        const auto intrinsic_events = std::make_shared<IntrinsicEvents>(symbol);
        const auto indicators = std::make_shared<Indicators>(symbol);

        const auto path = std::string{ BitBotLiveV1::path } + "/training_data";
        const auto timestamp_end = DateTime::now();
        const auto timestamp_start = timestamp_end - std::chrono::hours{ 365 * 24 };
        training_data.make(path, symbol, binance_klines, intrinsic_events, indicators, timestamp_start, timestamp_end);
    }
}
