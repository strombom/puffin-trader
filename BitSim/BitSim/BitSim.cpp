
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBaseClient.h"
#include "Logger.h"

#include <iostream>


int main() {
    logger.info("BitSim started");

    auto bitbase_client = BitBaseClient();

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } +std::chrono::minutes{ 0 };
    constexpr auto timestamp_end = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::seconds{ 30 };
    constexpr auto interval = std::chrono::seconds{ 10s };

    auto intervals = bitbase_client.get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval);

    //torch::Tensor tensor = torch::eye(5);
    //std::cout << tensor << std::endl;
}
