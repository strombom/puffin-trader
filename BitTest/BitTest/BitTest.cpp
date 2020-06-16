#include "pch.h"

#include "FeaturePlotter.h"
#include "BitLib/BitBaseClient.h"

#include <string>
#include <iostream>


int main()
{
    const auto task = std::string{ "plot_features" };

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
}
