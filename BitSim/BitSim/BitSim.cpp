#include "pch.h"

#include "BitBaseClient.h"
#include "BitBotConstants.h"
#include "BitmexSimulator.h"
#include "FE_Observations.h"
#include "FE_Inference.h"
#include "FE_Training.h"
#include "FE_Model.h"
#include "RL_Closer.h"
#include "Logger.h"
#include "Utils.h"

#include "DateTime.h"
#include <iostream>


int main()
{
    logger.info("BitSim started");

    //constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } +std::chrono::minutes{ 0 } +std::chrono::seconds{ 0 };
    //constexpr auto timestamp_end = date::sys_days(date::year{ 2020 } / 02 / 01) + std::chrono::hours{ 0 } +std::chrono::minutes{ 0 } +std::chrono::seconds{ 0 };

    constexpr auto command = "train_closer";

    if (command == "make_observations") {
        auto bitbase_client = BitBaseClient();
        constexpr auto symbol = "XBTUSD";
        constexpr auto exchange = "BITMEX";
        //constexpr auto interval = std::chrono::seconds{ 10s };
        auto intervals = bitbase_client.get_intervals(symbol, exchange, BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        intervals->save(BitSim::intervals_path);

        auto observations = std::make_shared<FE_Observations>( std::move(intervals), BitSim::timestamp_start );
        observations->save(BitSim::observations_path);
    }
    else if (command == "train") {
        auto observations = std::make_shared<FE_Observations>( BitSim::observations_path );

        auto fe_training = FE_Training{ observations };
        fe_training.train();
        fe_training.save_weights(BitSim::tmp_path, "fe_weights.pt");
    }
    else if (command == "inference") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        auto inference = FE_Inference{ BitSim::tmp_path, "fe_weights_0893.pt" };
        auto features = inference.forward(observations->get_all());

        std::cout << "Inference, features " << features.sizes() << std::endl;
        Utils::save_tensor(features, BitSim::tmp_path, "features.tensor");
    }
    else if (command == "train_closer") {
        auto intervals = std::make_shared<Intervals>(BitSim::intervals_path);
        auto features = Utils::load_tensor(BitSim::tmp_path, "features.tensor");
        std::cout << "features: " << features.sizes() << std::endl;

        auto simulator = std::make_shared<BitmexSimulator>(intervals);
        auto rl_closer = RL_Closer{ features, simulator };
        rl_closer.train();
    }
    
    logger.info("BitSim exit");
}
