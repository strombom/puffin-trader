#include "pch.h"

#include "FeaturePlotter.h"

#include <string>
#include <iostream>


int main()
{
    const auto task = std::string{ "plot_features" };

    if (task == "plot_features") {
        const auto features_path = "C:\\development\\github\\puffin-trader\\tmp";
        const auto features_filename = "features_1.tensor";
        auto feature_plotter = FeaturePlotter{ features_path, features_filename };

        const auto image_filename = "features.png";
        feature_plotter.plot(features_path, image_filename);

    }
}
