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
    bool has_prediction(const BitSim::Symbol& symbol);
    double get_prediction_score(const BitSim::Symbol& symbol, int delta_idx);

private:

    std::array<std::vector<Prediction>, BitSim::symbols.size()> data;
    std::array<int, BitSim::symbols.size()> data_idx;
    std::array<bool, BitSim::symbols.size()> active;

    void save(const BitSim::Symbol& symbol);
    bool load(const BitSim::Symbol& symbol);
    void load_csv(const BitSim::Symbol& symbol);
};
