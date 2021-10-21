#include "pch.h"
#include "Klines.h"
#include "Predictions.h"
#include "BitLib/BitBotConstants.h"

#include <iostream>


int main()
{
    const auto klines = Klines{};
    //const auto predictions = Predictions{};

    auto timestamp = klines.get_timestamp_start();
    const auto timestamp_end = klines.get_timestamp_end();

    while (timestamp < timestamp_end) {

        timestamp += std::chrono::minutes{ 1 };
    }

}
