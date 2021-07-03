#include "pch.h"

#include "BinanceDownloadKlines.h"

#include <iostream>


int main()
{
    logger.info("BitSim started");

    const auto command = std::string{ "download_klines" };

    if (command == "download_klines") {
        auto binance_download_klines = BinanceDownloadKlines{};
        binance_download_klines.download();

    }
}
