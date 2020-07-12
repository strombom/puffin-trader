#pragma once
#include "pch.h"

#include "Intervals.h"
#include "BitBotConstants.h"

#include <string>


class FE_Observations
{
public:
    FE_Observations(const std::string& file_path);
    FE_Observations(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals);

    void load(const std::string& file_path);
    void save(const std::string& file_path) const;

    void rotate_insert(sptrIntervals intervals, size_t new_intervals_count);

    size_t size(void);
    torch::Tensor get(int index);
    torch::Tensor get_all(void);
    torch::Tensor get(c10::ArrayRef<size_t> index);
    torch::Tensor get_random(int count);
    torch::Tensor get_range(int start, int count);
    torch::Tensor get_range(int start, int count, int step);
    torch::Tensor get_tail(int count);

    void print(void);

private:
    std::mutex get_mutex;

    torch::Tensor observations; // TxCxL (10000x6x32)
    time_point_ms timestamp_start;
    std::chrono::milliseconds interval;

    void calculate_observations(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals, size_t start_idx);
    void calculate_observation(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals, const std::vector<float>& binance_offsets, const std::vector<float>& coinbase_offsets, int obs_idx);
    float price_transform(float price);
    float bitmex_volume_transform(float volume, int feature_idx);
    float price_offset_transform(float volume, int feature_idx);
    float binance_volume_buy_transform(float volume, int feature_idx);
    float binance_volume_sell_transform(float volume, int feature_idx);
};

using sptrFE_Observations = std::shared_ptr<FE_Observations>;
