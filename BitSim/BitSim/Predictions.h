#pragma once
#include "Symbols.h"
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"

#include <array>
#include <filesystem>
#include <unordered_map>



struct Prediction
{
    time_point_ms timestamp;
    std::array<float, BitBot::Trading::delta_count> prediction;
    std::array<int, BitBot::Trading::delta_count> ground_truth;
};

class Predictions
{
public:
    Predictions(void);

    void step_idx(time_point_ms timestamp);
    bool has_prediction(const Symbol& symbol);
    double get_prediction_score(const Symbol& symbol, int delta_idx);

private:

    std::array<std::vector<Prediction>, symbols.size()> data;
    std::array<int, symbols.size()> data_idx;
    std::array<bool, symbols.size()> active;

    void save(const Symbol& symbol);
    bool load(const Symbol& symbol);
    void load_csv(const Symbol& symbol);
};
