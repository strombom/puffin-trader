#pragma once

#include "Intervals.h"

#include <string>

#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)


class FE_Observations
{
public:
    FE_Observations(sptrIntervals intervals, time_point_s start_time);

    void save(const std::string& file_path);

private:
    torch::Tensor observations; // TxCxL (10000x3x160)
    time_point_s start_time;
    std::chrono::seconds interval;

    torch::Tensor make_observation(sptrIntervals intervals, int idx_obs);
    float price_transform(float start_price, float price);
    float volume_transform(float volume);
};
