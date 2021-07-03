#include "pch.h"
#include "BinanceDownloadKlines.h"

#include "BitLib/BitBotConstants.h"


void BinanceDownloadKlines::download(void)
{
    for (const auto symbol : BitBot::symbols) {
        logger.info("symbol %s", symbol);
    }
}
