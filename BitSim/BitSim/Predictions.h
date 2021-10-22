#pragma once
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"

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
    Predictions(void);

    void step_idx(time_point_ms timestamp);

private:

    std::array<std::vector<Prediction>, BitBot::symbols.size()> data;
    std::array<int, BitBot::symbols.size()> data_idx;

    void save(const BitBot::Symbol& symbol);
    bool load(const BitBot::Symbol& symbol);
    void load_csv(const BitBot::Symbol& symbol);
};
