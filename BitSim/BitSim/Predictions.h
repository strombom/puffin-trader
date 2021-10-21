#pragma once
#include "BitLib/DateTime.h"

#include <filesystem>
#include <unordered_map>
#include <array>


constexpr int delta_count = 13;


struct Prediction
{
    time_point_ms timestamp;
    std::array<float, delta_count> prediction;
    std::array<int, delta_count> ground_truth;
};

class Predictions
{
public:
    Predictions();

private:
    void read_csv(const std::string symbol, const std::filesystem::path& path);

    std::unordered_map<std::string, std::vector<Prediction>> data;

    void save_predictions(const std::string symbol);
    bool load_predictions(const std::string symbol);
};
