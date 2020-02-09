#include "pch.h"

#include "BitBotConstants.h"
#include "BitBaseClient.h"
#include "Logger.h"
#include "FE_Observations.h"
#include "FE_Inference.h"
#include "FE_Training.h"
#include "FE_Model.h"
#include "Utils.h"

#include "DateTime.h"
#include <iostream>


int main()
{
    logger.info("BitSim started");

    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } +std::chrono::minutes{ 0 } +std::chrono::seconds{ 0 };
    constexpr auto timestamp_end = date::sys_days(date::year{ 2020 } / 02 / 01) + std::chrono::hours{ 0 } +std::chrono::minutes{ 0 } +std::chrono::seconds{ 0 };
    auto observations = sptrFE_Observations{ nullptr };

    constexpr auto command = "inference";

    if (command == "make_observations") {
        auto bitbase_client = BitBaseClient();
        constexpr auto symbol = "XBTUSD";
        constexpr auto exchange = "BITMEX";
        constexpr auto interval = std::chrono::seconds{ 10s };
        auto intervals = bitbase_client.get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval);
        observations = std::make_shared<FE_Observations>( std::move(intervals), timestamp_start );
        observations->save(BitSim::observations_path);
    }
    else if (command == "train") {
        observations = std::make_shared<FE_Observations>( BitSim::observations_path );

        auto fe_training = FE_Training{ observations };
        fe_training.train();
        fe_training.save_weights("C:\\development\\github\\puffin-trader\\tmp\\fe_weights.pt");
    }
    else if (command == "visual") {
        constexpr auto logdir = "C:\\development\\github\\puffin-trader\\tmp\\log";
    }
    else if (command == "inference") {
        observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        auto inference = FE_Inference{ "C:\\development\\github\\puffin-trader\\tmp\\fe_weights_0893.pt" };

        auto random_observations = observations->get_range(0, 3000, 64);
        auto features = inference.forward(random_observations);

        Utils::save_tensor(features, BitSim::tmp_path, "features.tensor");
    }
    
    logger.info("BitSim exit");
}
