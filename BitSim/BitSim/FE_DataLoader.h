#pragma once

#include "Intervals.h"
#include "BitSim.h"

#include <random>
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)


class RandomRange
{
    // https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
public:
    RandomRange(const int range_min, const int range_max) :
        random_generator(std::mt19937{ std::random_device{}() }),
        rand_int(range_min, range_max),
        range_min(range_min), range_max(range_max) {}

    RandomRange(const RandomRange& random_range) :
        random_generator(std::mt19937{ std::random_device{}() }),
        rand_int(random_range.range_min, random_range.range_max),
        range_min(random_range.range_min), range_max(random_range.range_max) {}

    int get(void) {
        return rand_int(random_generator);
    }

private:
    int range_min;
    int range_max;
    std::mt19937 random_generator;
    std::uniform_int_distribution<int> rand_int;
};

struct Batch {
    torch::Tensor past_observations;   // BxCxNxL (2x3x4x160)
    torch::Tensor future_positives;    // BxCxNxL (2x3x(1x1)x160)
    torch::Tensor future_negatives;    // BxCxNxL (2x3x(1x9)x160)

    Batch(const int batch_size, RandomRange* random_range, const Intervals& intervals) :
        past_observations(torch::empty({ batch_size, 3, BitSim::n_observations, BitSim::observation_length })),
        future_positives(torch::empty({ batch_size, 3, BitSim::n_predictions * BitSim::n_positive, BitSim::observation_length })),
        future_negatives(torch::empty({ batch_size, 3, BitSim::n_predictions * BitSim::n_negative, BitSim::observation_length }))
    {
        for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            const auto time_index = random_range->get();

            for (auto obs_idx = 0; obs_idx < BitSim::n_observations; ++obs_idx) {
                const auto first_time_idx = time_index - (BitSim::n_observations - obs_idx) * BitSim::observation_length;
                const auto first_price = intervals.rows[first_time_idx].last_price;

                for (auto feature_idx = 0; feature_idx < BitSim::observation_length; ++feature_idx) {
                    const auto row = &intervals.rows[(int)((long)first_time_idx + feature_idx)];
                    const auto price_log = std::log2f(row->last_price / first_price);

                    past_observations[batch_idx][BitSim::ch_price][obs_idx][feature_idx] = price_log;
                    past_observations[batch_idx][BitSim::ch_buy_volume][obs_idx][feature_idx] = row->vol_buy;
                    past_observations[batch_idx][BitSim::ch_sell_volume][obs_idx][feature_idx] = row->vol_sell;
                }
            }

            for (auto pred_idx = 0; pred_idx < BitSim::n_predictions * BitSim::n_positive; ++pred_idx) {
                const auto first_time_idx = time_index + pred_idx * BitSim::observation_length;
                const auto first_price = intervals.rows[first_time_idx].last_price;

                for (auto feature_idx = 0; feature_idx < BitSim::observation_length; ++feature_idx) {
                    const auto row = &intervals.rows[(int)((long)first_time_idx + feature_idx)];
                    const auto price_log = std::log2f(row->last_price / first_price);

                    future_positives[batch_idx][BitSim::ch_price][pred_idx][feature_idx] = price_log;
                    future_positives[batch_idx][BitSim::ch_buy_volume][pred_idx][feature_idx] = row->vol_buy;
                    future_positives[batch_idx][BitSim::ch_sell_volume][pred_idx][feature_idx] = row->vol_sell;
                }
            }

            for (auto pred_idx = 0; pred_idx < BitSim::n_predictions * BitSim::n_negative; ++pred_idx) {
                const auto random_index = random_range->get();
                const auto first_price = intervals.rows[random_index].last_price;

                for (auto feature_idx = 0; feature_idx < BitSim::observation_length; ++feature_idx) {
                    const auto row = &intervals.rows[(int)((long)random_index + feature_idx)];
                    const auto price_log = std::log2f(row->last_price / first_price);

                    future_negatives[batch_idx][BitSim::ch_price][pred_idx][feature_idx] = price_log;
                    future_negatives[batch_idx][BitSim::ch_buy_volume][pred_idx][feature_idx] = row->vol_buy;
                    future_negatives[batch_idx][BitSim::ch_sell_volume][pred_idx][feature_idx] = row->vol_sell;
                }
            }
        }

        //Utils::save_tensor(past_observations, "past_observations.tensor");
        //Utils::save_tensor(future_positives,  "future_positives.tensor");
        //Utils::save_tensor(future_negatives,  "future_negatives.tensor");
    }
};


class TradeDataset : public torch::data::BatchDataset<TradeDataset, Batch, c10::ArrayRef<size_t>>
{
public:
    TradeDataset(std::unique_ptr<Intervals> intervals) :
        intervals(*intervals),
        random_index(BitSim::n_observations* BitSim::feature_size, (int)intervals->rows.size() - BitSim::n_predictions * BitSim::feature_size) {}

    Batch get_batch(c10::ArrayRef<size_t> request) {
        const auto batch_size = (int)request.size();
        //const auto time_index = random_index.get();
        return Batch{ batch_size, &random_index, intervals };
    }

    c10::optional<size_t> size(void) const {
        return BitSim::batches_per_epoch * BitSim::batch_size;
    }

private:
    RandomRange random_index;
    Intervals intervals;

    //void reset(void) {}
    //void save(torch::serialize::OutputArchive& archive) const {}
    //void load(torch::serialize::InputArchive& archive) {}
};
