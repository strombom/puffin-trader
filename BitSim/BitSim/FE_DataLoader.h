#pragma once
#include "pch.h"

#include "FE_Observations.h"
#include "BitLib/BitBotConstants.h"
#include "BitLib/Utils.h"


struct Batch {
    torch::Tensor past_observations;   // BxCxNxL (2x3x4x128)
    torch::Tensor future_positives;    // BxCxNxL (2x3x(1x1)x128)
    torch::Tensor future_negatives;    // BxCxNxL (2x3x(1x9)x128)

    Batch(const int batch_size) :
        past_observations(torch::empty({ batch_size, BitSim::FeatureEncoder::n_channels, BitSim::n_observations,                     BitSim::FeatureEncoder::observation_length })),
        future_positives (torch::empty({ batch_size, BitSim::FeatureEncoder::n_channels, BitSim::n_predictions,                      BitSim::FeatureEncoder::observation_length })),
        future_negatives (torch::empty({ batch_size, BitSim::FeatureEncoder::n_channels, BitSim::n_predictions * BitSim::n_negative, BitSim::FeatureEncoder::observation_length })) {}
};


class TradeDataset : public torch::data::BatchDataset<TradeDataset, Batch, c10::ArrayRef<size_t>>
{
public:
    TradeDataset(sptrFE_Observations observations) :
        observations(observations),
        random_index(BitSim::n_observations * BitSim::feature_size, (int)observations->size() - BitSim::n_predictions * BitSim::feature_size) {}

    Batch get_batch(c10::ArrayRef<size_t> request);

    c10::optional<size_t> size(void) const;

    //void reset(void) {}
    //void save(torch::serialize::OutputArchive& archive) const {}
    //void load(torch::serialize::InputArchive& archive) {}

private:
    RandomRange random_index;
    sptrFE_Observations observations;
};
