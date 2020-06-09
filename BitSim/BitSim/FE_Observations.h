#pragma once
#include "pch.h"

#include "Intervals.h"

#include <string>


class FE_Observations
{
public:
    FE_Observations(const std::string& file_path);
    FE_Observations(sptrIntervals intervals);

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

    torch::Tensor observations; // TxCxL (10000x3x128)
    time_point_ms timestamp_start;
    std::chrono::milliseconds interval;

    void calculate_observations(sptrIntervals intervals, size_t start_idx);
    torch::Tensor calculate_observation(sptrIntervals intervals, int idx_obs);
    float price_transform(float start_price, float price);
    float volume_transform(float volume);
};

using sptrFE_Observations = std::shared_ptr<FE_Observations>;
