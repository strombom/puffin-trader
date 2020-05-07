#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitBase
{
    namespace Bitmex
    {
        constexpr auto exchange_name = "BITMEX"; 
        constexpr auto first_timestamp = time_point_s{ date::sys_days(date::year{2019} / 06 / 01) };

        namespace Daily
        {
            constexpr auto downloader_client_id = "bitmex_daily";
            constexpr auto base_url_start = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
            constexpr auto base_url_end = ".csv.gz";
            constexpr auto url_date_format = "%Y%m%d";
            constexpr auto active_downloads_max = 5;
        }
    }

    namespace Interval
    {
        constexpr auto enabled_symbols = std::array<const char*, 1>{ "XBTUSD" };
        constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
        constexpr auto batch_timeout = 1s;
        constexpr auto batch_size = 10000;
        constexpr auto intervals = std::array<std::chrono::seconds, 1>{ 10s };
    }

    namespace Database
    {
        constexpr auto time_format = "%F %T";
        constexpr auto root_path = "C:\\development\\github\\puffin-trader\\database";
        constexpr auto sqlite_busy_timeout_ms = 5000;
    }
}

namespace BitSim
{
    namespace BitBase
    {
        constexpr auto address = "tcp://localhost:31000";
    }

    constexpr auto observations_path = "C:\\development\\github\\puffin-trader\\tmp\\observations.dat";
    constexpr auto intervals_path = "C:\\development\\github\\puffin-trader\\tmp\\intervals.dat";
    constexpr auto tmp_path = "C:\\development\\github\\puffin-trader\\tmp";

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto interval = std::chrono::seconds{ 10s };
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + 0h + 0min + 0s;
    //constexpr auto timestamp_end = date::sys_days(date::year{ 2020 } / 02 / 01) + 0h + 0min + 0s;
    constexpr auto timestamp_end = date::sys_days(date::year{ 2019 } / 06 / 02) + 0h + 0min + 0s;

    constexpr auto n_batches = 20000;
    constexpr auto batch_size = 500;
    constexpr auto observation_length = 128;
    constexpr auto n_channels = 3;
    constexpr auto n_observations = 10;
    constexpr auto n_predictions = 1;
    constexpr auto n_positive = 1;
    constexpr auto n_negative = 19;

    constexpr auto feature_size = 128;

    const auto ch_price = 0;
    const auto ch_buy_volume = 1;
    const auto ch_sell_volume = 2;

    namespace BitMex
    {
        constexpr auto leverage = 100.0;
        constexpr auto taker_fee = 0.075 / 100;
        constexpr auto maker_fee = -0.025 / 100;
        constexpr auto maintenance_rate = 0.5 / 100;
        constexpr auto max_leverage = 10.0;
        constexpr auto order_hysteresis = 0.1;
    }

    namespace Trader
    {
        constexpr auto algorithm = "SAC";
        const auto device = torch::Device{ torch::kCPU };

        constexpr auto n_episodes = 250;
        constexpr auto save_period = 100;

        constexpr auto max_steps = 200;
        constexpr auto episode_length = 10h; // 2*7*24h;

        constexpr auto market_order_threshold = 0.97;
        constexpr auto limit_order_threshold = 0.7;
        constexpr auto order_hysteresis = 0.1;

        namespace SAC
        {
            constexpr auto update_interval = 1;

            constexpr auto batch_size = 4;
            constexpr auto buffer_size = 50000;
            constexpr auto initial_random_action = 1000;

            constexpr auto hidden_dim = 256;

            constexpr auto alpha = 0.2;
            constexpr auto gamma_discount = 0.99;
            constexpr auto soft_tau = 0.005;
            constexpr auto learning_rate = 3e-4;
            constexpr auto learning_rate_entropy = learning_rate;
            constexpr auto learning_rate_qf_1 = learning_rate;
            constexpr auto learning_rate_qf_2 = learning_rate;
            constexpr auto learning_rate_actor = learning_rate;
        }

        namespace PPO
        {
            constexpr auto batch_size = 32;
            constexpr auto buffer_size = 10 * max_steps;
            constexpr auto update_epochs = 5;
            constexpr auto update_batch_size = 200;

            constexpr auto hidden_dim = 128;
            constexpr auto clip_param = 0.2;
            constexpr auto max_grad_norm = 0.5;
            constexpr auto actor_learning_rate = 1e-4;
            constexpr auto critic_learning_rate = 3e-4;
            constexpr auto action_clamp = 5.0;
            constexpr auto gamma_discount = 0.95;
        }

        constexpr auto state_dim = 5; // 5; // feature_size + 1; // Features, leverage (-1 to +1)
        constexpr auto action_dim_discrete = 3;
        constexpr auto action_dim_continuous = 0;

        //constexpr auto log_names = std::array<const char*, 6>{ "total loss", "actor loss", "alpha loss", "qf1 loss", "qf2 loss", "episode score" };
        constexpr auto log_names = std::array<const char*, 6>{ "total loss", "pg loss", "value loss", "entropy mean", "approx kl", "" };
        constexpr auto log_path = "C:\\development\\github\\puffin-trader\\tmp\\trader_training.csv";
    }
}
