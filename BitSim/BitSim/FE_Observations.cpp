#include "pch.h"

#include "FE_Observations.h"
#include "BitBotConstants.h"
#include "DateTime.h"
#include "Logger.h"
#include "Utils.h"

#include <future>


FE_Observations::FE_Observations(const std::string& file_path)
{
    load(file_path);
}

FE_Observations::FE_Observations(sptrIntervals intervals) :
    interval(intervals->interval), timestamp_start(intervals->get_timestamp_start())
{
    const auto n_threads = std::max(1, (int)(std::thread::hardware_concurrency()) - 1);
    const auto n_observations = (long) intervals->rows.size() - (BitSim::observation_length - 1);
    if (n_observations <= 0) {
        observations = torch::empty({ 0, BitSim::n_channels, BitSim::observation_length });
    }
    else {
        observations = torch::empty({ n_observations, BitSim::n_channels, BitSim::observation_length });
        calculate_observations(intervals, 0);
    }
}

void FE_Observations::calculate_observations(sptrIntervals intervals, size_t start_idx)
{
    const auto n_observations = (int)size();
    const auto n_threads = std::max(1, (int)(std::thread::hardware_concurrency()) - 1);

    for (auto idx_obs = (int)start_idx; idx_obs < n_observations; idx_obs += n_threads) {
        auto futures = std::queue<std::future<torch::Tensor>>{};

        for (auto idx_task = 0; idx_task < n_threads && idx_obs + idx_task < n_observations; ++idx_task) {
            auto future = std::async(std::launch::async, &FE_Observations::calculate_observation, this, intervals, idx_obs + idx_task);
            futures.push(std::move(future));
        }

        for (auto idx_task = 0; !futures.empty(); ++idx_task) {
            observations[(long)idx_obs + idx_task] = futures.front().get();
            futures.pop();
        }

        if ((idx_obs - start_idx) % 100 == 0 && (idx_obs - start_idx) > 0) {
            logger.info("working %6.2f%%, %d / %d", ((float)(idx_obs - start_idx)) / (n_observations - start_idx) * 100, idx_obs - start_idx, n_observations - start_idx);
        }
    }
}

void FE_Observations::rotate_insert(sptrIntervals intervals, size_t new_intervals_count)
{
    if (new_intervals_count == 0) {
        return;
    }

    timestamp_start += interval * new_intervals_count;

    if (new_intervals_count < size()) {
        // Roll observations left to make place for new observations at the end
        assert(new_intervals_count < INT32_MAX);
        observations.roll(-(int)new_intervals_count, 0);
    }

    const auto first_new_idx = std::max(0, (int)size() - (int)new_intervals_count);
    calculate_observations(intervals, first_new_idx);
}

torch::Tensor FE_Observations::calculate_observation(sptrIntervals intervals, int idx_obs)
{
    auto observation = torch::empty({ BitSim::n_channels, BitSim::observation_length });
    const auto first_price = intervals->rows[idx_obs].last_price;

    for (auto idx_interval = 0; idx_interval < BitSim::observation_length; ++idx_interval) {
        const auto&& row = &intervals->rows[(int)((long)idx_obs + idx_interval)];
        observation[BitSim::ch_price][idx_interval] = price_transform(first_price, row->last_price);
        observation[BitSim::ch_buy_volume][idx_interval] = volume_transform(row->vol_buy);
        observation[BitSim::ch_sell_volume][idx_interval] = volume_transform(row->vol_sell);
    }

    return observation;
}

void FE_Observations::save(const std::string& file_path) const
{
    torch::save(observations, file_path + "_tensor");

    auto file = std::ofstream{ file_path + "_attr" };
    file << timestamp_start.time_since_epoch().count() << std::endl;
    file << interval.count() << std::endl;
    file.close();
}

void FE_Observations::load(const std::string& file_path)
{
    auto start_time_raw = 0;
    auto interval_raw = 0;

    {
        auto file = std::ifstream{ file_path + "_attr" };
        file >> start_time_raw >> interval_raw;
        file.close();
    }

    timestamp_start = time_point_ms{ std::chrono::milliseconds{start_time_raw} };
    interval = std::chrono::milliseconds{ interval_raw };

    torch::load(observations, file_path + "_tensor");
}

void FE_Observations::print(void)
{
    std::cout << "Observations, start time: " << DateTime::to_string(timestamp_start) << std::endl;
    std::cout << observations.sizes() << std::endl;
}

size_t FE_Observations::size(void)
{
    return observations.size(0);
}

torch::Tensor FE_Observations::get(int index)
{
    auto lock = std::scoped_lock(get_mutex);

    return observations[index];
}

torch::Tensor FE_Observations::get(c10::ArrayRef<size_t> indices)
{
    auto lock = std::scoped_lock(get_mutex);
    auto output = torch::empty({ (long)indices.size(), BitSim::n_channels, BitSim::feature_size });

    for (auto idx = 0; idx < indices.size(); ++idx) {
        output[idx] = observations[indices[idx]];
    }
    return output;
}

torch::Tensor FE_Observations::get_all(void)
{
    return observations;
}

torch::Tensor FE_Observations::get_random(int count)
{
    auto random = RandomRange{ 0, (int) observations.size(0) };
    auto indices = std::vector<size_t>{};
    for (auto idx = 0; idx < count; ++idx) {
        indices.push_back(random.get());
    }
    return get(indices);
}

torch::Tensor FE_Observations::get_range(int start, int count)
{
    return get_range(start, count, 1);
}

torch::Tensor FE_Observations::get_range(int start, int count, int step)
{
    auto indices = std::vector<size_t>{};
    for (auto idx = 0; idx < count; ++idx) {
        indices.push_back(start + idx * step);
    }
    return get(indices);
}

torch::Tensor FE_Observations::get_tail(int count)
{
    return get_range((int)size() - count, count);
}

float FE_Observations::price_transform(float start_price, float price)
{
    // Transform the price ratio into a -1 to 1 distribution
    auto indicator = price / start_price - 1.0f;
    auto sign = 1;
    if (indicator < 0) {
        sign = -1;
    }

    indicator = std::powf(sign * indicator, 0.01f);
    if (indicator > 0.902f) {
        indicator -= 0.902f;
    }
    indicator *= sign * 12.5f;

    return indicator;
}

float FE_Observations::volume_transform(float volume)
{
    // Transform the volume into a 0 to 1 distribution
    auto indicator = std::powf(volume, 0.1f);
    if (indicator > 0.95f) {
        indicator -= 0.95f;
    }
    indicator *= 0.2f;

    return indicator;
}
