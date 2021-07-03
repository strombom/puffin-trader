#include "pch.h"
#include "BinanceDownloadKlines.h"

#include "BitLib/BitBotConstants.h"
#include "BinanceRestApi.h"
#include "BinanceKlines.h"

#include <filesystem>


void BinanceDownloadKlines::download(void)
{
    for (const auto symbol : BitBot::symbols) {
        const auto file_path = std::string{ BitBot::Binance::Klines::path } + "\\" + symbol + ".dat";
        auto klines = std::make_shared<BinanceKlines>();

        if (std::filesystem::exists(file_path)) {
            klines = std::make_shared<BinanceKlines>(file_path);
        }

        auto start_timestamp = BitBot::first_timestamp;
        if (klines->rows.size() > 0)
        {
            start_timestamp = klines->get_timestamp_end();
        }

        while (true) {
            auto binance_rest_api = BinanceRestApi{};
            auto count = binance_rest_api.get_klines(symbol, start_timestamp, klines);
            break;
        }

        klines->save(file_path);
        logger.info("symbol %s", symbol);
    }
}
