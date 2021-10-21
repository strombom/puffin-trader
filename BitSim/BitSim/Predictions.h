#pragma once
#include "BitLib/DateTime.h"

#include <array>
#include <filesystem>
#include <unordered_map>


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

    void save(const std::string symbol);
    bool load(const std::string symbol);
};
