#pragma once

#include "DateTime.h"

#include <array>

using namespace std::chrono_literals;

namespace BitBase
{
    namespace Bitmex
    {
        constexpr auto exchange_name = "BITMEX"; 
        constexpr auto first_timestamp = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) + std::chrono::hours{ 0 } };
        constexpr auto symbols = std::array<const char*, 1>{ "XBTUSD" };

        namespace Daily
        {
            constexpr auto downloader_client_id = "bitmex_daily";
            constexpr auto base_url_start = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/";
            constexpr auto base_url_end = ".csv.gz";
            constexpr auto url_date_format = "%Y%m%d";
            constexpr auto active_downloads_max = 5;
        }

        namespace Live
        {
            constexpr auto address = "tcp://delta.superdator.se:31002";
            constexpr auto max_rows = 100000;
        }
        
        namespace Interval
        {
            constexpr auto steps = std::array<float, 6>{ 1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 50.0f };
            constexpr auto batch_timeout = 1s;
            constexpr auto batch_size = 10000;
            constexpr auto intervals = std::array<std::chrono::milliseconds, 1>{ 10s };
        }
    }

    namespace Binance
    {
        constexpr auto exchange_name = "BINANCE";
        constexpr auto first_timestamp = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) + std::chrono::hours{ 0 } };
        constexpr auto symbols = std::array<const char*, 1>{ "BTCUSDT" };

        namespace Live
        {
            constexpr auto address = "tcp://delta.superdator.se:31003";
            constexpr auto max_rows = 10000;

            constexpr auto api_key = "N2T7yLPKHiMGA2zJEcc4gI0AxJAxCMO7YqVGDJqQa6uF3PdqjZvHRM2oqkplitps";
            constexpr auto api_secret = "RqqXIVn3Q9W3asDUXZt2jpJIrJMQ4ALOicZYNppkxPHy8pDsR5nCTG9P5YX9FIew";
            constexpr auto rest_api_auth_timeout = 60s;
            constexpr auto rest_api_host = "api.binance.com";
            constexpr auto rest_api_port = "443";
            constexpr auto rest_api_url = "/api/v3/";

            constexpr auto rate_limit = 100ms;
        }

        namespace Interval
        {
            constexpr auto steps = std::array<float, 0>{};
            constexpr auto batch_timeout = 1s;
            constexpr auto batch_size = 10000;
            constexpr auto intervals = std::array<std::chrono::milliseconds, 1>{ 10s };
        }
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
    namespace LiveData
    {
        constexpr auto intervals_buffer_length = 22min; // 4h;
    }

    namespace BitBase
    {
        constexpr auto address = "tcp://localhost:31001";
        constexpr auto interval = 10s;
    }

    //constexpr auto feature_encoder_weights_filename = "fe_weights_20200524e.pt";
    constexpr auto feature_encoder_weights_filename = "fe_weights_20200618.pt";
    constexpr auto policy_weights_filename = "model_9999_policy_jan_aprl_2h.net";
    constexpr auto observations_path = "C:\\development\\github\\puffin-trader\\tmp\\observations.dat";
    constexpr auto intervals_path = "C:\\development\\github\\puffin-trader\\tmp\\intervals.dat";
    constexpr auto tmp_path = "C:\\development\\github\\puffin-trader\\tmp";

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto interval = std::chrono::milliseconds{ 10s };
    constexpr auto timestamp_start = date::sys_days(date::year{ 2020 } / 1 / 1) + 0h + 0min + 0s;
    constexpr auto timestamp_end = date::sys_days(date::year{ 2020 } / 5 / 1) + 0h + 0min + 0s;

    constexpr auto n_batches = 2000;
    constexpr auto batch_size = 500;
    constexpr auto observation_length = 8; // Adjust FeatureEncoder
    constexpr auto n_channels = 1;
    constexpr auto n_observations = 5;
    constexpr auto n_predictions = 1;
    constexpr auto n_negative = 10;

    constexpr auto feature_size = 8;

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
        namespace Bitmex
        {
            constexpr auto websocket_host = "www.bitmex.com";
            constexpr auto websocket_port = "443";
            constexpr auto websocket_url = "/realtime";
            constexpr auto api_key = "ynOrYOWoC1knanjDld9RtPhC";
            constexpr auto api_secret = "0d_jDIPan7mEHSPhQDyMQJKVPJ3kEc5qbS5ed5JBWiKIsAXW";
            constexpr auto websocket_auth_timeout = 10000h;
            constexpr auto rest_api_auth_timeout = 60s;
            constexpr auto rest_api_host = "www.bitmex.com";
            constexpr auto rest_api_port = "443";
            constexpr auto rest_api_url = "/api/v1/";
        }

#ifdef TORCH_API
        const auto device = torch::Device{ torch::kCUDA };
#endif

        constexpr auto algorithm = "SAC";

        constexpr auto n_episodes = 10000;
        constexpr auto save_period = 50;

        constexpr auto max_steps = 2000;
        constexpr auto episode_length = 2h;

        constexpr auto order_hysteresis = 0.1;

        namespace SAC
        {
            constexpr auto update_interval = n_episodes / 200;

            constexpr auto batch_size = 512;
            constexpr auto buffer_size = 50000;
            constexpr auto initial_random_action = 1000;

            constexpr auto hidden_dim = 2048; // 4096;

            constexpr auto alpha = 1.0;
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

        constexpr auto state_dim = feature_size + 1;
        constexpr auto action_dim_discrete = 6;
        constexpr auto action_dim_continuous = 0;

        constexpr auto log_names = std::array<const char*, 6>{ "total loss", "pg loss", "value loss", "entropy mean", "approx kl", "" };
        constexpr auto log_path = "C:\\development\\github\\puffin-trader\\tmp\\trader_training.csv";
    }
}
