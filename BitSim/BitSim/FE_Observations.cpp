#include "pch.h"

#include "FE_Observations.h"
#include "DateTime.h"
#include "Logger.h"
#include "Utils.h"

#include <future>


FE_Observations::FE_Observations(const std::string& file_path)
{
    load(file_path);
}

FE_Observations::FE_Observations(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals) :
    interval(bitmex_intervals->interval), timestamp_start(bitmex_intervals->get_timestamp_start())
{
    const auto n_observations = (long)bitmex_intervals->rows.size() - (BitSim::FeatureEncoder::observation_length - 1);
    if (n_observations <= 0) {
        observations = torch::empty({ 0, BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::feature_length });
    }
    else {
        observations = torch::empty({ n_observations, BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::feature_length });
        calculate_observations(std::move(bitmex_intervals), std::move(binance_intervals), std::move(coinbase_intervals), 0);
    }
}

void FE_Observations::calculate_observations(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals, size_t start_idx)
{
    const auto n_observations = (int)size();

    std::cout << "Observations " << observations.sizes() << std::endl;

    auto binance_offsets = std::vector<float>(BitSim::intervals_length);
    auto coinbase_offsets = std::vector<float>(BitSim::intervals_length);

    auto binance_diff_ema = binance_intervals->rows[0].last_price - bitmex_intervals->rows[0].last_price;
    auto coinbase_diff_ema = coinbase_intervals->rows[0].last_price - bitmex_intervals->rows[0].last_price;
    binance_offsets[0] = binance_diff_ema;
    coinbase_offsets[0] = coinbase_diff_ema;

    for (auto idx = 1; idx < BitSim::intervals_length; ++idx) {
        const auto binance_diff = binance_intervals->rows[idx].last_price - bitmex_intervals->rows[idx].last_price;
        const auto coinbase_diff = coinbase_intervals->rows[idx].last_price - bitmex_intervals->rows[idx].last_price;

        binance_diff_ema = BitSim::FeatureEncoder::offset_ema_alpha * binance_diff + (1 - BitSim::FeatureEncoder::offset_ema_alpha) * binance_diff_ema;
        binance_offsets[idx] = (binance_intervals->rows[idx].last_price - bitmex_intervals->rows[idx].last_price - (float)(binance_diff_ema));
        coinbase_diff_ema = BitSim::FeatureEncoder::offset_ema_alpha * coinbase_diff + (1 - BitSim::FeatureEncoder::offset_ema_alpha) * coinbase_diff_ema;
        coinbase_offsets[idx] = (coinbase_intervals->rows[idx].last_price - bitmex_intervals->rows[idx].last_price - (float)(coinbase_diff_ema));
    }

    /*
    const auto filename = "C:\\development\\github\\puffin-trader\\tmp\\direction\\observations.csv";
    {
        auto csv = CSVLogger{ {"bitmex_price",  "binance_offsets", "coinbase_offsets"}, filename };
        for (auto idx = 0; idx < BitSim::intervals_length; ++idx) {

            csv.append_row({
                (double)bitmex_intervals->rows[idx].last_price,
                (double)binance_offsets[idx],
                (double)coinbase_offsets[idx]
                });
        }
    }
    */

    const auto n_threads = 1; // std::max(1, (int)(std::thread::hardware_concurrency()) - 1);

    /*
    auto buffer_tensors = std::vector<torch::Tensor>{};
    auto buffer_accessors = std::vector<at::TensorAccessor<float, 2>>{};

    for (auto thread_idx = 0; thread_idx < n_threads; ++thread_idx) {
        auto tensor = torch::empty({ BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::feature_length });
        buffer_tensors.push_back(tensor);
        buffer_accessors.push_back(tensor.accessor<float, 2>());
    }
    */

    for (auto idx_obs = (int)start_idx; idx_obs < n_observations; idx_obs += n_threads) {
        auto futures = std::queue<std::future<void>>{};
        const auto idx_task = 0;
        calculate_observation(bitmex_intervals, binance_intervals, coinbase_intervals, binance_offsets, coinbase_offsets, idx_obs + idx_task);
        /*
        for (auto idx_task = 0; idx_task < n_threads && idx_obs + idx_task < n_observations; ++idx_task) {
            auto future = std::async(std::launch::async, &FE_Observations::calculate_observation, this, buffer_accessors[idx_task], bitmex_intervals, binance_intervals, coinbase_intervals, binance_offsets, coinbase_offsets, idx_obs + idx_task);
            futures.push(std::move(future));
        }

        for (auto idx_task = 0; !futures.empty(); ++idx_task) {
            futures.front().wait();

            //auto observation = observations.narrow(0, idx_obs + idx_task, 1).squeeze();
            observations.narrow(0, idx_obs + idx_task, 1).squeeze() = buffer_tensors[idx_task];

            futures.pop();
        }
        */

        //if ((idx_obs - start_idx) % 100 == 0 && (idx_obs - start_idx) > 0) {
        if (idx_obs % 1000 == 0) {
            logger.info("working %6.2f%%, %d / %d", ((float)(idx_obs - start_idx)) / (n_observations - start_idx) * 100, idx_obs - start_idx, n_observations - start_idx);
        }
    }
    
    //std::cout << "obs " << observations << std::endl;
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
    calculate_observations(intervals, intervals, intervals, first_new_idx);
}

void FE_Observations::calculate_observation(sptrIntervals bitmex_intervals, sptrIntervals binance_intervals, sptrIntervals coinbase_intervals, const std::vector<float>& binance_offsets, const std::vector<float>& coinbase_offsets, int obs_idx)
{
    auto interval_idx = obs_idx + BitSim::FeatureEncoder::observation_length - 1;
    const auto bitmex_first_price = bitmex_intervals->rows[interval_idx].last_price;
    interval_idx -= 1;
    auto observation = observations.narrow(0, obs_idx, 1).squeeze();
    auto obs_access = observation.accessor<float, 2>();

    for (auto feature_idx = 0; feature_idx < BitSim::FeatureEncoder::feature_length; ++feature_idx) {
        const auto end_idx = obs_idx + BitSim::FeatureEncoder::observation_length - 1 - BitSim::FeatureEncoder::lookback_index[feature_idx];

        auto bitmex_max_price = -std::numeric_limits<float>::max();
        auto bitmex_min_price = std::numeric_limits<float>::max();
        auto bitmex_volume_buy = 0.0f;
        auto bitmex_volume_sell = 0.0f;

        auto binance_max_offset = -std::numeric_limits<float>::max();
        auto binance_min_offset = std::numeric_limits<float>::max();
        auto binance_volume_buy = 0.0f;
        auto binance_volume_sell = 0.0f;

        auto coinbase_max_offset = -std::numeric_limits<float>::max();
        auto coinbase_min_offset = std::numeric_limits<float>::max();
        auto coinbase_volume_buy = 0.0f;
        auto coinbase_volume_sell = 0.0f;

        for (; interval_idx >= end_idx; --interval_idx) {
            const auto bitmex_price = bitmex_intervals->rows[interval_idx].last_price - bitmex_first_price;
            bitmex_max_price = std::max(bitmex_max_price, bitmex_price);
            bitmex_min_price = std::min(bitmex_min_price, bitmex_price);
            bitmex_volume_buy += bitmex_intervals->rows[interval_idx].vol_buy;
            bitmex_volume_sell += bitmex_intervals->rows[interval_idx].vol_sell;

            const auto binance_offset = binance_offsets[interval_idx];
            binance_max_offset = std::max(binance_max_offset, binance_offset);
            binance_min_offset = std::min(binance_min_offset, binance_offset);
            binance_volume_buy += binance_intervals->rows[interval_idx].vol_buy;
            binance_volume_sell += binance_intervals->rows[interval_idx].vol_sell;

            const auto coinbase_offset = coinbase_offsets[interval_idx];
            coinbase_max_offset = std::max(coinbase_max_offset, coinbase_offset);
            coinbase_min_offset = std::min(coinbase_min_offset, coinbase_offset);
            coinbase_volume_buy += coinbase_intervals->rows[interval_idx].vol_buy;
            coinbase_volume_sell += coinbase_intervals->rows[interval_idx].vol_sell;
        }

        bitmex_volume_buy /= BitSim::FeatureEncoder::lookback_length[feature_idx];
        bitmex_volume_sell /= BitSim::FeatureEncoder::lookback_length[feature_idx];
        binance_volume_buy /= BitSim::FeatureEncoder::lookback_length[feature_idx];
        binance_min_offset /= BitSim::FeatureEncoder::lookback_length[feature_idx];
        coinbase_volume_buy /= BitSim::FeatureEncoder::lookback_length[feature_idx];
        coinbase_volume_sell /= BitSim::FeatureEncoder::lookback_length[feature_idx];

        if (feature_idx < 2 && bitmex_max_price > 4000) {
            std::cout << "was obs(" << obs_idx << ") maxprice(" << bitmex_max_price << ")" << std::endl;
        }
        obs_access[0][feature_idx] = price_transform(bitmex_max_price);
        obs_access[1][feature_idx] = price_transform(bitmex_min_price);
        obs_access[2][feature_idx] = bitmex_volume_transform(bitmex_volume_buy, feature_idx);
        obs_access[3][feature_idx] = bitmex_volume_transform(bitmex_volume_sell, feature_idx);
        obs_access[4][feature_idx] = binance_price_offset_transform(binance_max_offset, feature_idx);
        obs_access[5][feature_idx] = binance_price_offset_transform(binance_min_offset, feature_idx);
        obs_access[6][feature_idx] = binance_volume_buy_transform(binance_volume_buy, feature_idx);
        obs_access[7][feature_idx] = binance_volume_sell_transform(binance_volume_sell, feature_idx);
        obs_access[8][feature_idx] = coinbase_price_offset_transform(coinbase_max_offset, feature_idx);
        obs_access[9][feature_idx] = coinbase_price_offset_transform(coinbase_min_offset, feature_idx);
        obs_access[10][feature_idx] = coinbase_volume_buy_transform(coinbase_volume_buy, feature_idx);
        obs_access[11][feature_idx] = coinbase_volume_sell_transform(coinbase_volume_sell, feature_idx);
    }
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
    auto start_time_raw = 0ll;
    auto interval_raw = 0ll;

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
    auto output = torch::empty({ (long)indices.size(), BitSim::FeatureEncoder::n_channels, BitSim::feature_size });

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

float FE_Observations::price_transform(float price)
{
    // Transform the relative price into a -1 to 1 distribution
    const auto sign = price >= 0 ? 1.0f : -1.0f;
    auto indicator = std::powf(sign * price, 0.001f);
    if (indicator > 0.998f) {
        indicator -= 0.998f;
    }
    return sign * indicator * 125;
}

float FE_Observations::bitmex_volume_transform(float volume, int feature_idx)
{
    constexpr auto a = std::array<float, BitSim::FeatureEncoder::feature_length>{
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.15f, 0.06f, 0.06f, 0.06f, 0.06f, 0.06f
    };

    constexpr auto b = std::array<float, BitSim::FeatureEncoder::feature_length>{
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 4.0f, 5.0f, 6.0f
    };

    constexpr auto c = std::array<float, BitSim::FeatureEncoder::feature_length>{
        0.85f, 0.85f, 0.85f, 0.85f, 0.85f, 0.85f, 0.85f, 0.9f, 1.0f, 1.2f, 1.3f, 3.1f, 3.1f, 3.3f, 4.2f, 5.2f
    };

    // Transform the volume into a [-1, 1] distribution
    const auto max = std::powf(1.45f, 16.0f - feature_idx) * 60000;
    const auto indicator = std::powf(volume / max, a[feature_idx]) * b[feature_idx] - c[feature_idx];
    return indicator;
}

float FE_Observations::binance_price_offset_transform(float price_offset, int feature_idx)
{
    // Transform the offset into a [-1, 1] distribution
    const auto sign = price_offset >= 0 ? 1.0f : -1.0f;
    const auto indicator = sign * std::powf(sign * price_offset, 0.25f) * 0.5f;
    return indicator;
}

float FE_Observations::binance_volume_buy_transform(float volume, int feature_idx)
{
    // Transform the volume into a [-1, 1] distribution
    const auto indicator = std::powf(volume, 0.1f)
        * (2.0f + (BitSim::FeatureEncoder::feature_length - 1) / 15.0f)
        - (1.5f + feature_idx / 15.0f);
    return indicator;
}

float FE_Observations::binance_volume_sell_transform(float volume, int feature_idx)
{
    // Transform the volume into a [-1, 1] distribution
    const auto indicator = std::powf(volume, 0.1f)
        * (2.0f + (BitSim::FeatureEncoder::feature_length - 1) / 15.0f)
        - (0.5f + std::powf(1.12f, (float)feature_idx));
    return indicator;
}

float FE_Observations::coinbase_price_offset_transform(float price_offset, int feature_idx)
{
    // Transform the offset into a [-1, 1] distribution
    const auto sign = price_offset >= 0 ? 1.0f : -1.0f;
    const auto indicator = sign * std::powf(sign * price_offset, 0.25f)
        * 0.5f
        - (0.6f + 0.3f * feature_idx / (BitSim::FeatureEncoder::feature_length - 1));
    return indicator;
}

float FE_Observations::coinbase_volume_buy_transform(float volume, int feature_idx)
{
    // Transform the volume into a [-1, 1] distribution
    const auto indicator = std::powf(volume, 0.1f)
        * (2.0f + 1.5f * feature_idx / (BitSim::FeatureEncoder::feature_length - 1))
        - (1.3f + 0.8f * feature_idx / (BitSim::FeatureEncoder::feature_length - 1));
    return indicator;
}

float FE_Observations::coinbase_volume_sell_transform(float volume, int feature_idx)
{
    // Transform the volume into a [-1, 1] distribution
    const auto indicator = std::powf(volume, 0.1f)
        * (1.0f + 2.5f * feature_idx / (BitSim::FeatureEncoder::feature_length - 1))
        - (1.3f - 2.2f * feature_idx / (BitSim::FeatureEncoder::feature_length - 1));
    return indicator;
}
