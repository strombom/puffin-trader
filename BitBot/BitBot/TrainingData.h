#pragma once
#include "pch.h"

#include "BinanceKlines.h"
#include "Indicators.h"



class TrainingData
{
public:
    TrainingData();

    void make(std::string_view path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end);
    //void make_section(const std::string& path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end);
    void make_sections(std::string_view path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start);

    void join(void);

private:
    void make_ground_truth(std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators);

    struct Position;
    std::shared_ptr<std::vector<std::array<int, BitBot::Trading::take_profit.size()>>> ground_truth;
    std::shared_ptr<std::vector<std::array<time_point_ms, BitBot::Trading::take_profit.size()>>> ground_truth_timestamps;
    std::string ground_truth_symbol;

    void save_ground_truth(std::string_view symbol);
    void load_ground_truth(std::string_view symbol);
    
    std::vector<std::thread> threads;
};

struct TrainingData::Position
{
public:
    Position(int ind_idx, float take_profit, float stop_loss) :
        ind_idx(ind_idx), take_profit(take_profit), stop_loss(stop_loss), remove(false) {}

    int ind_idx;
    float take_profit;
    float stop_loss;
    bool remove;
};
