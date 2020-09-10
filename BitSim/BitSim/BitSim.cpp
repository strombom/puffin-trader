#include "pch.h"

#include "BitmexTrader.h"
#include "FE_Observations.h"
#include "FE_Inference.h"
#include "FE_Training.h"
#include "FE_Model.h"
#include "MT_Evaluator.h"
#include "RL_Trader.h"
#include "PD_Events.h"
#include "PD_Simulator.h"
#include "LiveData.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib\BitBaseClient.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"
#include "BitLib/Utils.h"
#include "BitLib/AggTicks.h"

#include <iostream>


int main()
{
    logger.info("BitSim started");

    const auto command = std::string{ "train_rl" };

    if (command == "download_ticks") {
        auto bitbase_client = BitBaseClient();
        auto bitmex_ticks = bitbase_client.get_ticks("XBTUSD", "BITMEX", BitSim::timestamp_start, BitSim::timestamp_end);
        bitmex_ticks->save(std::string{ BitSim::tmp_path } + "\\bitmex_ticks.dat");

    }
    else if (command == "aggregate_ticks") {
        auto bitmex_ticks = std::make_shared<Ticks>(std::string{ BitSim::tmp_path } + "\\bitmex_ticks.dat");
        auto bitmex_agg_ticks = AggTicks{ bitmex_ticks };
        bitmex_agg_ticks.save(std::string{ BitSim::tmp_path } + "\\bitmex_agg_ticks.dat");
    }
    else if (command == "find_direction_changes") {
        auto ticks = std::make_shared<Ticks>(std::string{ BitSim::tmp_path } + "\\bitmex_ticks.dat");
        auto pd_events = PD_Events{ ticks };

        auto bitbase_client = BitBaseClient();
        auto bitmex_intervals = bitbase_client.get_intervals("XBTUSD", "BITMEX", BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        pd_events.plot_events(bitmex_intervals);
    }
    else if (command == "get_intervals") {
        auto bitbase_client = BitBaseClient();
        auto bitmex_intervals = bitbase_client.get_intervals("XBTUSD", "BITMEX", BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        auto binance_intervals = bitbase_client.get_intervals("BTCUSDT", "BINANCE", BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        auto coinbase_intervals = bitbase_client.get_intervals("BTC-USD", "COINBASE_PRO", BitSim::timestamp_start, BitSim::timestamp_end, BitSim::interval);
        std::cout << "Intervals: " << bitmex_intervals->rows.size() << std::endl;
        bitmex_intervals->save(std::string{ BitSim::intervals_path } + "_bitmex.dat");
        binance_intervals->save(std::string{ BitSim::intervals_path } + "_binance.dat");
        coinbase_intervals->save(std::string{ BitSim::intervals_path } + "_coinbase.dat");
    }
    else if (command == "make_observations") {
        auto bitmex_intervals = sptrIntervals{};
        auto binance_intervals = sptrIntervals{};
        auto coinbase_intervals = sptrIntervals{};
        bitmex_intervals->load(std::string{ BitSim::intervals_path } + "_bitmex.dat");
        binance_intervals->load(std::string{ BitSim::intervals_path } + "_binance.dat");
        coinbase_intervals->load(std::string{ BitSim::intervals_path } + "_coinbase.dat");
        auto observations = std::make_shared<FE_Observations>(bitmex_intervals, binance_intervals, coinbase_intervals);
        observations->save(BitSim::observations_path);
        std::cout << "Observations: " << observations->get_all().sizes() << std::endl;
    }
    else if (command == "train_feature_encoder") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);

        auto fe_training = FE_Training{ observations };
        fe_training.train();
        fe_training.save_weights(BitSim::tmp_path, "fe_weights.pt");
    }
    else if (command == "make_features") {
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        std::cout << "Observations: " << observations->get_all().sizes() << std::endl;

        auto inference = FE_Inference{ BitSim::tmp_path, BitSim::feature_encoder_weights_filename };
        auto features = inference.forward(observations->get_all());
        std::cout << "Inference, features " << features.sizes() << std::endl;

        Utils::save_tensor(features, BitSim::tmp_path, "features.tensor");
    }
    else if (command == "train_mt") {
        auto ticks = std::make_shared<Ticks>(std::string{ BitSim::tmp_path } + "\\bitmex_ticks.dat");

        auto evaluator = MT_Evaluator{ ticks };
        evaluator.evaluate();
    }
    else if (command == "train_rl") {
        auto bitmex_agg_ticks = sptrAggTicks{};
        bitmex_agg_ticks->load(std::string{ BitSim::tmp_path } + "\\bitmex_agg_ticks.dat");

        auto simulator = std::make_shared<PD_Simulator>(bitmex_agg_ticks);

        //auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        //auto intervals = std::make_shared<Intervals>(BitSim::intervals_path);
        //auto features = Utils::load_tensor(BitSim::tmp_path, "features.tensor");
        //std::cout << "observations: " << observations->get_all().sizes() << std::endl;
        //std::cout << "features: " << features.sizes() << std::endl;
        //std::cout << "intervals: " << intervals->rows.size() << std::endl;

        //auto features = observations->get_all().flatten(1).cpu();

        //auto simulator = std::make_shared<BitmexSimulator>(ticks);
        //auto rl_trader = RL_Trader{ simulator };
        //rl_trader.train();
    }
    else if (command == "evaluate_rl") {
        /*
        auto observations = std::make_shared<FE_Observations>(BitSim::observations_path);
        auto intervals = std::make_shared<Intervals>(BitSim::intervals_path);
        auto features = Utils::load_tensor(BitSim::tmp_path, "features.tensor");
        std::cout << "observations: " << observations->get_all().sizes() << std::endl;
        std::cout << "features: " << features.sizes() << std::endl;
        std::cout << "intervals: " << intervals->rows.size() << std::endl;

        auto simulator = std::make_shared<PD_Simulator>(intervals, features.cpu());
        auto rl_trader = RL_Trader{ simulator };
        //rl_trader.train();
        const auto idx_episode = 0;
        const auto timestamp_start = date::sys_days(date::year{ 2020 } / 4 / 1) + 0h + 0min + 0s;
        const auto timestamp_end = date::sys_days(date::year{ 2020 } / 4 / 1) + 2h + 0min + 0s;
        rl_trader.evaluate(idx_episode, timestamp_start, timestamp_end);
        */
    }
    else if (command == "trade_live") {
        auto live_data = std::make_shared<LiveData>();
        auto rl_policy = std::make_shared<RL_Policy>(BitSim::policy_weights_filename);
        auto bitmex_trader = BitmexTrader{ live_data, rl_policy };

        live_data->start();
        bitmex_trader.start();

        //std::this_thread::sleep_for(1s);
        while (true) {
            //break;
            auto command = std::string{};
            std::cin >> command;
            if (command.compare("q") == 0) {
                break;
            }
        }

        bitmex_trader.shutdown();
        live_data->shutdown();
    }
    
    logger.info("BitSim exit");
}
