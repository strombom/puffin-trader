#include "pch.h"

#include "FE_Observations.h"

#include "BitBotConstants.h"
#include "DateTime.h"
#include "Logger.h"

#include <future>


FE_Observations::FE_Observations(const std::string& file_path)
{
    load(file_path);
}

FE_Observations::FE_Observations(sptrIntervals intervals, time_point_s start_time) :
    start_time(start_time), interval(intervals->interval)
{
    const auto n_threads = std::max(1, (int)(std::thread::hardware_concurrency()) - 1);
    const auto n_observations = (long) intervals->rows.size() - BitSim::observation_length + 1;
    if (n_observations <= 0) {
        observations = torch::empty({ 0, BitSim::n_channels, BitSim::observation_length });
        return;
    }
    observations = torch::empty({ n_observations, BitSim::n_channels, BitSim::observation_length });

    for (auto i = 0; i < 1; ++i) {
        auto timer = Timer{};

        for (auto idx_obs = 0; idx_obs < n_observations; idx_obs += n_threads) {
            auto futures = std::queue<std::future<torch::Tensor>>{};

            for (auto idx_task = 0; idx_task < n_threads && idx_obs + idx_task < n_observations; ++idx_task) {
                auto future = std::async(std::launch::async, &FE_Observations::make_observation, this, intervals, idx_obs + idx_task);
                futures.push(std::move(future));
            }

            for (auto idx_task = 0; !futures.empty(); ++idx_task) {
                auto a = futures.front().get();
                observations[(long)idx_obs + idx_task] = a;
                //std::cout << a << std::endl;
                futures.pop();
            }

            if (idx_obs % 100 == 0) {
                logger.info("working %6.2f%%, %d / %d", ((float)idx_obs) / n_observations * 100, idx_obs, n_observations);
            }
        }

        timer.print_elapsed("done");
    }
}

torch::Tensor FE_Observations::make_observation(sptrIntervals intervals, int idx_obs)
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

void FE_Observations::save(const std::string& file_path)
{
    torch::save(observations, file_path + "_tensor");

    auto file = std::ofstream{ file_path + "_attr" };
    file << start_time.time_since_epoch().count() << std::endl;
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

    start_time = time_point_s{ std::chrono::seconds{start_time_raw} };
    interval = std::chrono::seconds{ interval_raw };

    torch::load(observations, file_path + "_tensor");
}

void FE_Observations::print(void)
{
    std::cout << "Observations, start time: " << datetime_to_string(start_time) << std::endl;
    std::cout << observations.sizes() << std::endl;
}

int64_t FE_Observations::size(void)
{
    return observations.size(0);
}

torch::Tensor FE_Observations::get(int index)
{
    auto lock = std::scoped_lock(get_mutex);

    return observations[index];
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
