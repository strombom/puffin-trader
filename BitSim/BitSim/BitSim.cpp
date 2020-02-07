#include "pch.h"

#include "BitBotConstants.h"
#include "BitBaseClient.h"
#include "Logger.h"
#include "FE_Observations.h"
#include "FE_Inference.h"
#include "FE_Training.h"
#include "FE_Model.h"

#include "DateTime.h"
#include <iostream>


int main()
{
    /*
    auto obs = torch::zeros({ 2, 2, 3 });
    auto batch = torch::zeros({ 2, 2, 2, 3 });

    obs[0][0][0] = 1;
    obs[0][0][1] = 2;
    obs[0][0][2] = 3;
    obs[0][1][0] = 4;
    obs[0][1][1] = 5;
    obs[0][1][2] = 6;

    obs[1][0][0] = 2;
    obs[1][0][1] = 3;
    obs[1][0][2] = 4;
    obs[1][1][0] = 5;
    obs[1][1][1] = 6;
    obs[1][1][2] = 7;

    //batch[0].slice(2, 0, 0, 1) = obs[0];

    std::cout << "one batch" << std::endl;
    std::cout << batch[0] << std::endl;

    std::cout << "slice" << std::endl;
    std::cout << batch[0].slice(1, 0, 1, 1).reshape({2, 3}) << std::endl;

    std::cout << "one obs" << std::endl;
    std::cout << obs[0] << std::endl;

    batch[0].slice(1, 0, 1, 1).reshape({ 2, 3 }) = obs[0];
    batch[0].slice(1, 1, 2, 1).reshape({ 2, 3 }) = obs[1];

    std::cout << "obs" << std::endl;
    std::cout << obs << std::endl;

    std::cout << "batch" << std::endl;
    std::cout << batch << std::endl;

    std::cout << "one batch" << std::endl;
    std::cout << batch[0].slice(1, 0, 1, 1).reshape({ 2, 3 }) << std::endl;
    return 1;
    */

    //batch.past_observations[batch_idx].slice(2, obs_idx, obs_idx, 1) = observations->get(obs_time_idx);

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
        //auto a = torch::tensorb
        constexpr auto logdir = "C:\\development\\github\\puffin-trader\\tmp\\log";
    }
    else if (command == "inference") {
        observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        auto inference = FE_Inference{ "C:\\development\\github\\puffin-trader\\tmp\\fe_weights_0893.pt" };

    }
    
    //observations->print();    

    logger.info("BitSim exit");
}
