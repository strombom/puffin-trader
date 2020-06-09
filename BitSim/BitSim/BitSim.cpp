#include "pch.h"

#include "BitBaseClient.h"
#include "BitBotConstants.h"
#include "BitmexTrader.h"
#include "BitmexSimulator.h"
#include "FE_Observations.h"
#include "FE_Inference.h"
#include "FE_Training.h"
#include "FE_Model.h"
#include "RL_Trader.h"
#include "LiveData.h"
#include "DateTime.h"
#include "Logger.h"
#include "Utils.h"

#include <iostream>


int main()
{
    logger.info("BitSim started");

    constexpr auto command = "trade_live";

    if (command == "make_observations") {
        auto bitbase_client = BitBaseClient();
        constexpr auto symbol = "XBTUSD";
        constexpr auto exchange = "BITMEX";
        constexpr auto interval = std::chrono::seconds{ 10s };
        auto intervals = bitbase_client.get_intervals(symbol, exchange, BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        std::cout << "Intervals: " << intervals->rows.size() << std::endl;
        intervals->save(BitSim::intervals_path);

        auto observations = std::make_shared<FE_Observations>(intervals);
        observations->save(BitSim::observations_path);
        std::cout << "Observations: " << observations->get_all().sizes() << std::endl;
    }
    else if (command == "train_features") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);

        auto fe_training = FE_Training{ observations };
        fe_training.train();
        fe_training.save_weights(BitSim::tmp_path, "fe_weights.pt");
    }
    else if (command == "inference") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        std::cout << "Observations: " << observations->get_all().sizes() << std::endl;

        auto inference = FE_Inference{ BitSim::tmp_path, "fe_weights_20200523.pt" };
        auto features = inference.forward(observations->get_all());
        std::cout << "Inference, features " << features.sizes() << std::endl;

        Utils::save_tensor(features, BitSim::tmp_path, "features.tensor");
    }
    else if (command == "train_rl") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        auto intervals = std::make_shared<Intervals>(BitSim::intervals_path);
        auto features = Utils::load_tensor(BitSim::tmp_path, "features.tensor");
        std::cout << "observations: " << observations->get_all().sizes() << std::endl;
        std::cout << "features: " << features.sizes() << std::endl;
        std::cout << "intervals: " << intervals->rows.size() << std::endl;

        auto simulator = std::make_shared<BitmexSimulator>(intervals, features.cpu());
        auto rl_trader = RL_Trader{ simulator };
        rl_trader.train();
    }
    else if (command == "trade") {
        auto bitmex_trader = BitmexTrader{};
        bitmex_trader.start();
        while (true) {
            auto command = std::string{};
            std::cin >> command;
            if (command.compare("q") == 0) {
                break;
            }
        }
        bitmex_trader.shutdown();
    }
    else if (command == "trade_live") {
        auto live_data = LiveData{};
        live_data.start();
        //std::this_thread::sleep_for(1s);
        while (true) {
            //break;
            auto command = std::string{};
            std::cin >> command;
            if (command.compare("q") == 0) {
                break;
            }
        }
        live_data.shutdown();
    }
    
    logger.info("BitSim exit");
}
