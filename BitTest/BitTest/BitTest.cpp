#include "pch.h"

#include "FeaturePlotter.h"
#include "DirectionClient.h"
#include "BitLib/BitBaseClient.h"

#include <string>
#include <iostream>


int main()
{
    const auto task = std::string{ "direction_data" };

    if (task == "plot_features") {
        auto bitbase_client = BitBaseClient{};

        auto intervals = bitbase_client.get_intervals(
            "XBTUSD", 
            "BITMEX",
            time_point_ms{ date::sys_days(date::year{2020} / 4 / 30) + std::chrono::hours{ 0 } },
            time_point_ms{ date::sys_days(date::year{2020} / 5 / 2) + std::chrono::hours{ 0 } },
            10s
            );

        auto size = intervals->rows.size();
        auto prices = std::vector<double>{};
        prices.reserve(size);
        for (auto row : intervals->rows) {
            prices.push_back(row.last_price);
        }

        const auto features_path = "C:\\development\\github\\puffin-trader\\tmp";
        const auto features_filename = "features.tensor";
        auto feature_plotter = FeaturePlotter{ features_path, features_filename };

        const auto image_filename = "features.png";
        feature_plotter.plot(prices, features_path, image_filename);

    }
    else if (task == "plot_prices") {
        auto bitbase_client = BitBaseClient{};

        const auto timestamp_start = time_point_ms{ date::sys_days(date::year{ 2020 } / 5 / 1) + std::chrono::hours{ 0 } };
        const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2020} / 5 / 1) + std::chrono::minutes{ 1 } };
        const auto interval = 2s;

        const auto bitmex_intervals = bitbase_client.get_intervals("XBTUSD", "BITMEX", timestamp_start, timestamp_end, interval);

        auto timestamp = timestamp_start;
        for (auto row : bitmex_intervals->rows) {
            std::cout << DateTime::to_string_iso_8601(timestamp) << " - " << row.last_price << std::endl;
            timestamp += interval;
        }
    }
    else if (task == "direction_data") {
        const auto timestamp_start = time_point_ms{ date::sys_days(date::year{ 2020 } / 5 / 1) + std::chrono::hours{ 0 } };
        const auto timestamp_end = time_point_ms{ date::sys_days(date::year{2020} / 5 / 1) + std::chrono::minutes{ 2 } };
        const auto interval = 2s;

        auto bitbase_client = BitBaseClient{};
        const auto bitmex_intervals = bitbase_client.get_intervals("XBTUSD", "BITMEX", timestamp_start, timestamp_end, interval);
        const auto binance_intervals = bitbase_client.get_intervals("BTCUSDT", "BINANCE", timestamp_start, timestamp_end, interval);
        const auto coinbase_intervals = bitbase_client.get_intervals("BTC-USD", "COINBASE_PRO", timestamp_start, timestamp_end, interval);

        const auto length = bitmex_intervals->rows.size();

        // Make orderbook
        auto orderbook = std::vector<double>{};
        orderbook.resize(length);
        orderbook[0] = bitmex_intervals->rows[0].last_price;
        for (auto idx = 1; idx < length; ++idx) {
            if (bitmex_intervals->rows[idx].last_price > orderbook[idx - 1]) {
                orderbook[idx] = bitmex_intervals->rows[idx].last_price;
            }
            else if (bitmex_intervals->rows[idx].last_price < orderbook[idx - 1] - 0.5) {
                orderbook[idx] = bitmex_intervals->rows[idx].last_price + 0.5;
            }
            else {
                orderbook[idx] = orderbook[idx - 1];
            }
        }

        // Set directions
        //  0 - No direction
        //  1 - Buy
        //  2 - Sell
        auto directions = std::vector<int>{};
        directions.resize(length);
        for (auto idx = 1; idx < length; ++idx) {
            if (bitmex_intervals->rows[idx].last_price > orderbook[idx - 1]) {
                directions[idx - 1] = 2;
            }
            else if (bitmex_intervals->rows[idx].last_price < orderbook[idx - 1] - 0.5) {
                directions[idx - 1] = 1;
            }
        }

        auto last_direction = directions.back();
        for (int idx = length - 1; idx > -1; --idx) {
            if (directions[idx] == 0) {
                directions[idx] = last_direction;
            }
            else {
                last_direction = directions[idx];
            }
        }

        auto direction_client = DirectionClient{};
        direction_client.send({{"bitmex", bitmex_intervals}, {"binance", binance_intervals}, {"coinbase", coinbase_intervals}}, directions);
    }
}
