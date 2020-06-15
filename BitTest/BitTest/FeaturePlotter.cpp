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
    //generate some image
    const auto width = 1000;
    const auto height = 128;

    const auto local_features = features.cpu();
    const auto start_idx = local_features.size(0) - width - 1;

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

    unsigned error = lodepng::encode(file_path + "\\" + filename, image, width, height);

    //if there's an error, display it
    if (error) {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}


/*
//generate some image
const auto width = 4000;
const auto height = 128;
auto image = std::vector<unsigned char>{};
image.resize(width* height * 4);
auto tensor = Utils::load_tensor(BitSim::tmp_path, "features.tensor").cpu();
for (auto y = 0; y < height; y++) {
    for (auto x = 0; x < width; x++) {
        auto c = (unsigned int)(tensor[2108033 - width + x][0][y].item().toDouble() * 255);

        image[4 * width * y + 4 * x + 0] = c;
        image[4 * width * y + 4 * x + 1] = c;
        image[4 * width * y + 4 * x + 2] = c;
        image[4 * width * y + 4 * x + 3] = c;
    }
}
std::cout << "Tensor " << tensor.sizes() << std::endl;

const auto filename = "C:\\development\\github\\puffin-trader\\tmp\\features2.png";
//std::vector<unsigned char> image;

unsigned error = lodepng::encode(filename, image, width, height);

//if there's an error, display it
if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

return 0;

*/
