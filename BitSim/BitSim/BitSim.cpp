
#include "BitBaseClient.h"
#include "FE_Observations.h"
#include "FE_Training.h"
#include "DateTime.h"
#include "Logger.h"


int main() {
    logger.info("BitSim started");
    
    auto bitbase_client = BitBaseClient();

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } + std::chrono::minutes{ 0 };
    constexpr auto timestamp_end = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 24 };
    constexpr auto interval = std::chrono::seconds{ 10s };

    auto intervals = bitbase_client.get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval);
    
    auto observations = FE_Observations{ std::move(intervals), timestamp_start };
    observations.save(BitSim::observations_path);

    //auto fe_training = FE_Training{ std::move(intervals) };
    //fe_training.train();

    logger.info("BitSim exit");
}
