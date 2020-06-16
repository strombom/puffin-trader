#include "pch.h"

#include "FeaturePlotter.h"
#include "BitLib/Utils.h"

#include <lodepng.h>


FeaturePlotter::FeaturePlotter(const std::string& file_path, const std::string& filename)
{
    features = Utils::load_tensor(file_path, filename);
}

void FeaturePlotter::plot(std::vector<double> prices, const std::string& file_path, const std::string& filename)
{
    const auto local_features = features.cpu();


    assert(prices.size() == img_width);

    const auto price_height = 256;
    const auto price_max = *std::max_element(prices.begin(), prices.end());
    const auto price_min = *std::min_element(prices.begin(), prices.end());

    const auto features_start_idx = 259200; // local_features.size(0) - width - 1;
    const auto features_height = (int) local_features.size(2);

    const auto img_width = 17280;// 10000;
    const auto img_height = price_height + features_height;

    auto image = std::vector<unsigned char>{};
    image.resize(img_width * img_height * 4);
    std::fill(image.begin(), image.end(), 0);

    for (auto y = 0; y < features_height; y++) {
        for (auto x = 0; x < img_width; x++) {
            const auto c = (unsigned int)(local_features[features_start_idx + x][0][y].item().toDouble() * 255);
            image[4 * img_width * y + 4 * x + 0] = c;
            image[4 * img_width * y + 4 * x + 1] = c;
            image[4 * img_width * y + 4 * x + 2] = c;
            image[4 * img_width * y + 4 * x + 3] = 255;
        }
    }

    for (auto x = 0; x < img_width; x++) {
        for (auto y = 0; y < price_height; y+=10) {
            image[4 * img_width * (y + features_height) + 4 * x + 0] = 100;
            image[4 * img_width * (y + features_height) + 4 * x + 1] = 100;
            image[4 * img_width * (y + features_height) + 4 * x + 2] = 0;
            image[4 * img_width * (y + features_height) + 4 * x + 3] = 255;
        }
    }

    for (auto x = 0; x < img_width; x += 10) {
        for (auto y = 0; y < price_height; y++) {
            image[4 * img_width * (y + features_height) + 4 * x + 0] = 100;
            image[4 * img_width * (y + features_height) + 4 * x + 1] = 100;
            image[4 * img_width * (y + features_height) + 4 * x + 2] = 0;
            image[4 * img_width * (y + features_height) + 4 * x + 3] = 255;
        }
    }

    for (auto x = 0; x < img_width; x++) {
        const auto c = 255;
        const auto y = (int)(features_height + price_height * (1 - (prices[x] - price_min) / (price_max - price_min)) - 1);

        image[4 * img_width * y + 4 * x + 0] = c;
        image[4 * img_width * y + 4 * x + 1] = c;
        image[4 * img_width * y + 4 * x + 2] = c;
        image[4 * img_width * y + 4 * x + 3] = 255;
    }

    for (auto y = 0; y < img_height; y++) {
        const auto x = img_width / 2;
        image[4 * img_width * y + 4 * x + 0] = image[4 * img_width * y + 4 * x + 0] / 2 + 100;
        image[4 * img_width * y + 4 * x + 1] = image[4 * img_width * y + 4 * x + 1] / 2;
        image[4 * img_width * y + 4 * x + 2] = image[4 * img_width * y + 4 * x + 2] / 2;
        image[4 * img_width * y + 4 * x + 3] = 255;
    }

    const auto error = lodepng::encode(file_path + "\\" + filename, image, img_width, img_height);
    if (error) {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}
