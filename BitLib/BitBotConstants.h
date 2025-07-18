#pragma once

#include "DateTime.h"

#include <array>

using namespace std::string_literals;


namespace BitBotLiveV1
{
    constexpr auto path = "C:\\BitBotLiveV1";
    //constexpr auto history_length = std::chrono::hours{ 182 * 24 };
}

namespace BitSim
{
    /*
    struct Symbol {
        constexpr Symbol(int idx, const std::string_view name, float tick_size, float taker_fee, float maker_fee) : 
            idx(idx), name(name), tick_size(tick_size), taker_fee(taker_fee), maker_fee(maker_fee) {}

        const int idx;
        const std::string_view name;
        const float tick_size;
        const float taker_fee;
        const float maker_fee;
    };

    constexpr const auto symbols = std::array{
        Symbol{ 0, "ADAUSDT", 0.0001, 0.00075, -0.00025 },
        //Symbol{ 1, "BCHUSDT" }
    };
    */

    constexpr auto fee = -0.00025;
    constexpr auto min_position_value = 50;

    namespace Portfolio
    {
        constexpr auto total_capacity = 4;
        constexpr auto symbol_capacity = 2;
    }

    namespace Klines
    {
        constexpr auto path = "E:/BitBot/klines/";
    }

    namespace Predictions
    {
        constexpr auto path = "E:/BitBot/predictions/";
    }
}

namespace BitBot
{
    constexpr auto start_timestamp = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) + std::chrono::hours{ 0 } };
    //constexpr auto end_timestamp = time_point_ms{ date::sys_days(date::year{2021} / 7 / 5) + std::chrono::hours{ 0 } };
    //constexpr auto n_timestamps = std::chrono::duration_cast<std::chrono::minutes>(end_timestamp - start_timestamp).count();
    constexpr auto history_length = std::chrono::hours{ 30 * 24 }; //std::chrono::hours{ 182 * 24 };
    constexpr auto validation_length = date::days{ 1 };

    constexpr auto symbols = std::array<const char*, 21>{ "ADAUSDT", "ATOMUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT", "CHZUSDT", "DOGEUSDT", "EOSUSDT", "ETCUSDT", "ETHUSDT", "FTMUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "NEOUSDT", "THETAUSDT", "TRXUSDT", "VETUSDT", "XLMUSDT", "XRPUSDT" };
    //constexpr auto symbols = std::array<const char*, 16>{ "ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT", "CHZUSDT", "DOGEUSDT", "EOSUSDT", "ETHUSDT", "ETCUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "THETAUSDT", "XLMUSDT", "XRPUSDT" };
    //constexpr auto symbols = std::array<const char*, 12>{ "ADAUSDT", "BCHUSDT", "BNBUSDT", "BTTUSDT", "CHZUSDT", "EOSUSDT", "ETCUSDT", "LINKUSDT", "MATICUSDT", "THETAUSDT", "XLMUSDT", "XRPUSDT" };

    constexpr auto path = "E:/BitBot";

    namespace Klines
    {
    }

    namespace IntrinsicEvents
    {
        constexpr auto target_event_count = 250000;
    }

    namespace Indicators
    {
        constexpr auto degrees = std::array<int, 3>{1, 2, 3};
        constexpr auto lengths = std::array<int, 10>{5, 7, 11, 15, 22, 33, 47, 68, 100, 150};
        constexpr auto indicator_width = degrees.size() * lengths.size() * 2;

        constexpr auto max_degree = degrees.back();
        constexpr auto max_length = lengths.back();
        constexpr auto max_degree_p1 = max_degree + 1;
        constexpr auto max_degree_p2 = max_degree + 2;
        constexpr auto max_degree_t2p1 = max_degree * 2 + 1;
    }

    namespace Trading
    {
        //constexpr auto take_profit = std::array<double, 7>{1.008, 1.010, 1.012, 1.015, 1.018, 1.022, 1.027};
        //constexpr auto stop_loss   = std::array<double, 7>{0.992, 0.990, 0.988, 0.985, 0.982, 0.978, 0.973};
        constexpr auto delta_count = 15;
        constexpr auto take_profit = std::array<double, delta_count>{1.004, 1.005, 1.006, 1.007, 1.008, 1.009, 1.010, 1.011, 1.012, 1.013, 1.014, 1.015, 1.016, 1.017, 1.018};
        constexpr auto stop_loss   = std::array<double, delta_count>{0.996, 0.995, 0.994, 0.993, 0.992, 0.991, 0.990, 0.989, 0.988, 0.987, 0.986, 0.985, 0.984, 0.983, 0.982};
    }
}

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
            constexpr auto intervals = std::array<std::chrono::milliseconds, 0>{ };
        }
    }

    namespace Binance
    {
        constexpr auto exchange_name = "BINANCE";
        constexpr auto first_timestamp = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) + std::chrono::hours{ 0 } };
        constexpr auto symbols = std::array<const char*, 1>{ "BTCUSDT" };

        namespace Tick
        {
            constexpr auto api_key = "N2T7yLPKHiMGA2zJEcc4gI0AxJAxCMO7YqVGDJqQa6uF3PdqjZvHRM2oqkplitps";
            constexpr auto api_secret = "RqqXIVn3Q9W3asDUXZt2jpJIrJMQ4ALOicZYNppkxPHy8pDsR5nCTG9P5YX9FIew";
            constexpr auto rest_api_auth_timeout = 60s;
            constexpr auto rest_api_host = "api.binance.com";
            constexpr auto rest_api_port = "443";
            constexpr auto rest_api_url = "/api/v3/";

            constexpr auto rate_limit = 1000ms;
            constexpr auto max_rows = 1000;
        }

        namespace Live
        {
            constexpr auto buffer_length = 6h;
            constexpr auto address = "tcp://delta.superdator.se:31003";
            constexpr auto max_rows = 10000;
        }

        namespace Interval
        {
            constexpr auto steps = std::array<float, 0>{};
            constexpr auto batch_timeout = 1s;
            constexpr auto batch_size = 10000;
            constexpr auto intervals = std::array<std::chrono::milliseconds, 0>{ };
        }
    }

    namespace CoinbasePro
    {
        constexpr auto exchange_name = "COINBASE_PRO";
        constexpr auto first_timestamp = time_point_ms{ date::sys_days(date::year{2020} / 1 / 1) + std::chrono::hours{ 0 } };
        constexpr auto symbols = std::array<const char*, 1>{ "BTC-USD" };

        namespace Tick
        {
            constexpr auto api_key = "bbaf349a94d91ff0a84f12121e0bcf15";
            constexpr auto api_secret = "wj+LdtE7im6JJAHFZZfp9hd2YRT9rA0jqhocuHICz9y7LPgN66NlmDrM3hM84i2Jf+yvyuw2qJ3rmRpRaxX85Q==";
            constexpr auto api_passphrase = "zmkcltqij1c";
            constexpr auto rest_api_auth_timeout = 60s;
            constexpr auto rest_api_host = "api.pro.coinbase.com";
            constexpr auto rest_api_port = "443";
            constexpr auto rest_api_url = "/";

            constexpr auto rate_limit = 250ms;
            constexpr auto max_rows = 100;
            constexpr auto first_id = 80350317;
        }

        namespace Live
        {
            constexpr auto buffer_length = 12h;
            constexpr auto address = "tcp://delta.superdator.se:31004";
            constexpr auto max_rows = 10000;
        }

        namespace Interval
        {
            constexpr auto steps = std::array<float, 0>{};
            constexpr auto batch_timeout = 1s;
            constexpr auto batch_size = 10000;
            constexpr auto intervals = std::array<std::chrono::milliseconds, 3>{ 250ms, 500ms, 1s };
        }
    }

    namespace Database
    {
        constexpr auto time_format = "%F %T";
        //constexpr auto root_path = "C:\\development\\github\\puffin-trader\\database";
        constexpr auto root_path = "E:\\BitBase";
        constexpr auto sqlite_busy_timeout_ms = 5000;
    }
}

/*
namespace BitSim
{
    constexpr auto interval = 1000ms;
    constexpr auto aggregate = 2000ms;

    namespace LiveData
    {
        constexpr auto intervals_buffer_length = 22min; // 4h;
    }

    namespace BitBase
    {
        constexpr auto address = "tcp://localhost:31001";
    }

    namespace FeatureEncoder
    {
        constexpr auto feature_length = 16;
        constexpr auto lookback_index = std::array<int, feature_length>{ {1, 2, 3, 5, 11, 23, 45, 86, 164, 311, 590, 1119, 2120, 4016, 7604, 14399} };
        constexpr auto lookback_length = std::array<int, feature_length>{ {1, 1, 1, 2, 6, 12, 22, 41, 78, 147, 279, 529, 1001, 1896, 3588, 6795} };
        constexpr auto observation_length = lookback_index.back() + 1;
        constexpr auto observation_timespan = std::chrono::milliseconds{ observation_length * BitSim::interval.count() };
        constexpr auto offset_ema_alpha = 0.0001f;
        constexpr auto n_channels = 3 * 4;
    }

    constexpr auto timestamp_start = date::sys_days(date::year{ 2020 } / 1 / 1) + 0h + 0min + 0s;
    constexpr auto timestamp_end = date::sys_days(date::year{ 2021 } / 2 / 27) + 0h + 0min + 0s;
    constexpr auto intervals_length = (timestamp_end - timestamp_start) / interval;

    //constexpr auto feature_encoder_weights_filename = "fe_weights_20200524e.pt";
    constexpr auto feature_encoder_weights_filename = "fe_weights_20200618.pt";
    constexpr auto policy_weights_filename = "model_9999_policy_jan_aprl_2h.net";
    constexpr auto observations_path = "C:\\development\\github\\puffin-trader\\tmp\\observations.dat";
    constexpr auto intervals_path = "C:\\development\\github\\puffin-trader\\tmp\\intervals";
    constexpr auto tmp_path = "C:\\development\\github\\puffin-trader\\tmp";

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";

    constexpr auto n_batches = 2000;
    constexpr auto batch_size = 500;
    constexpr auto n_observations = 5;
    constexpr auto n_predictions = 1;
    constexpr auto n_negative = 10;

    constexpr auto feature_size = FeatureEncoder::feature_length * FeatureEncoder::n_channels; //  8;

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

    namespace PriceDirection
    {
        constexpr auto deltas = std::array<double, 7>{ {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} };
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

        namespace Mech
        {
            constexpr auto take_profit = 0.003;
            constexpr auto stop_loss = 0.0015;
            constexpr auto min_leverage_take_profit = 1.5;
            constexpr auto min_leverage_stop_loss = 4.5;
            constexpr auto volatility_buffer_length = 250;
        }

#ifdef TORCH_API
        const auto device = torch::Device{ torch::kCUDA };
#endif

        constexpr auto algorithm = "SAC";

        constexpr auto n_episodes = 10000;
        constexpr auto save_period = 50;

        constexpr auto max_steps = 2000;
        constexpr auto episode_length = 4h;

        constexpr auto order_hysteresis = 0.1;

        constexpr auto feature_events_count = 6;
        constexpr auto stop_loss_range = 0.01;

        namespace SAC
        {
            constexpr auto update_interval = 50; // episode_length / interval / 20;

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

        constexpr auto state_dim = 3 + feature_events_count * 2; // feature_size + 3;
        constexpr auto action_dim_discrete = 2;
        constexpr auto action_dim_continuous = 1;

        constexpr auto loss_log_names = std::array<const char*, 6>{ "total loss", "pg loss", "value loss", "entropy mean", "approx kl", "" };
        constexpr auto loss_log_path = "C:\\development\\github\\puffin-trader\\tmp\\trader_training.csv";

        constexpr auto episode_log_names = std::array<const char*, 7>{ "mark price", "timestamp", "account value", "position price", "position direction", "position stop loss", "position leverage" };
        constexpr auto episode_log_path = "C:\\development\\github\\puffin-trader\\tmp\\rl\\episode_log";
    }
}
*/
