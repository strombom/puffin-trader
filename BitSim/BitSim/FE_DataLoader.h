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

    Batch(const int batch_size) :
        past_observations(torch::empty({ batch_size, 3, BitSim::n_observations,                     BitSim::observation_length })), //.cuda()
        future_positives (torch::empty({ batch_size, 3, BitSim::n_predictions * BitSim::n_positive, BitSim::observation_length })),
        future_negatives (torch::empty({ batch_size, 3, BitSim::n_predictions * BitSim::n_negative, BitSim::observation_length })) {}
};


class TradeDataset : public torch::data::BatchDataset<TradeDataset, Batch, c10::ArrayRef<size_t>>
{
public:
    TradeDataset(std::unique_ptr<Intervals> intervals) :
        intervals(*intervals),
        random_index(BitSim::n_observations* BitSim::feature_size, (int)intervals->rows.size() - BitSim::n_predictions * BitSim::feature_size) {}

    Batch get_batch(c10::ArrayRef<size_t> request);

    c10::optional<size_t> size(void) const;

    //void reset(void) {}
    //void save(torch::serialize::OutputArchive& archive) const {}
    //void load(torch::serialize::InputArchive& archive) {}

private:
    RandomRange random_index;
    Intervals intervals;
};
