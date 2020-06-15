#include "pch.h"

#include "FeaturePlotter.h"
#include "Utils.h"

#include <lodepng.h>


FeaturePlotter::FeaturePlotter(const std::string& file_path, const std::string& filename)
{
    features = Utils::load_tensor(file_path, filename);
}

void FeaturePlotter::plot(const std::string& file_path, const std::string& filename)
{
    const auto local_features = features.cpu();

    const auto width = 10000;
    const auto start_idx = local_features.size(0) - width - 1;
    const auto height = (int) local_features.size(2);

    auto image = std::vector<unsigned char>{};
    image.resize(width * height * 4);
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            const auto c = (unsigned int)(local_features[start_idx + x][0][y].item().toDouble() * 255);
            image[4 * width * y + 4 * x + 0] = c;
            image[4 * width * y + 4 * x + 1] = c;
            image[4 * width * y + 4 * x + 2] = c;
            image[4 * width * y + 4 * x + 3] = c;
        }
    }

    const auto error = lodepng::encode(file_path + "\\" + filename, image, width, height);
    if (error) {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}
