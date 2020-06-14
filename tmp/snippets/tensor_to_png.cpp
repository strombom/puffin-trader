
    //generate some image
    const auto width = 4000;
    const auto height = 128;
    auto image = std::vector<unsigned char>{};
    image.resize(width * height * 4);
    auto tensor = Utils::load_tensor(BitSim::tmp_path, "features.tensor").cpu();
    for (auto y = 0; y < height; y++) {
        for (auto x = 0; x < width; x++) {
            auto c = (unsigned int) (tensor[2108033 - width + x][0][y].item().toDouble() * 255);
            
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