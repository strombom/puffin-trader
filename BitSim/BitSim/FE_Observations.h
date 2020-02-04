#pragma once
#include "pch.h"

#include "Intervals.h"

#include <string>


class FE_Observations
{
public:
    FE_Observations(const std::string& file_path);
    FE_Observations(sptrIntervals intervals, time_point_s start_time);

    void save(const std::string& file_path);
    void load(const std::string& file_path);

    int64_t size(void);
    torch::Tensor get(int index);

    void print(void);

private:
    torch::Tensor observations; // TxCxL (10000x3x128)
    time_point_s start_time;
    std::chrono::seconds interval;

    torch::Tensor make_observation(sptrIntervals intervals, int idx_obs);
    float price_transform(float start_price, float price);
    float volume_transform(float volume);
};

using sptrFE_Observations = std::shared_ptr<FE_Observations>;
