#include "pch.h"

#include "BinanceDownloadKlines.h"
#include "BinanceRestApi.h"
#include "BinanceKlines.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


void BinanceDownloadKlines::download(void)
{
    for (const auto symbol : BitBot::symbols) {
        logger.info("Downloading %s", symbol);

        auto klines = std::make_shared<BinanceKlines>(symbol);

        auto start_timestamp = BitBot::start_timestamp;
        while (true) {
            if (klines->rows.size() > 0) {
                start_timestamp = klines->get_timestamp_end() + std::chrono::minutes{ 1 };
            }

            auto binance_rest_api = BinanceRestApi{};
            auto count = binance_rest_api.get_klines(symbol, start_timestamp, klines);
            if (count < 1) {
                break;
            }
            logger.info("Downloading %s %s", symbol, DateTime::to_string_iso_8601(start_timestamp).c_str());
        }

        klines->save();
    }
}
