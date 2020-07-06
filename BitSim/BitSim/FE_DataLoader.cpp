#include "pch.h"

#include "FE_DataLoader.h"
#include "Utils.h"
#include "DateTime.h"


Batch TradeDataset::get_batch(c10::ArrayRef<size_t> request)
{
    //auto timer = Timer();

    const auto batch_size = (int)request.size();
    auto batch = Batch{ batch_size };

    for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto time_index = random_index.get();

        for (auto obs_idx = 0; obs_idx < BitSim::n_observations; ++obs_idx) {
            const auto obs_time_idx = time_index - (BitSim::n_observations - obs_idx) * BitSim::FeatureEncoder::observation_length;
            batch.past_observations[batch_idx].slice(1, obs_idx, obs_idx + 1, 1).reshape(c10::IntArrayRef{ {BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::observation_length} }) = observations->get(obs_time_idx);
        }

        for (auto pred_idx = 0; pred_idx < BitSim::n_predictions; ++pred_idx) {
            const auto obs_time_idx = time_index + pred_idx * BitSim::FeatureEncoder::observation_length;
            batch.future_positives[batch_idx].slice(1, pred_idx, pred_idx + 1, 1).reshape(c10::IntArrayRef{ {BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::observation_length} }) = observations->get(obs_time_idx);
        }

        for (auto neg_idx = 0; neg_idx < BitSim::n_predictions * BitSim::n_negative; ++neg_idx) {
            const auto obs_time_idx = random_index.get();
            batch.future_negatives[batch_idx].slice(1, neg_idx, neg_idx + 1, 1).reshape(c10::IntArrayRef{ {BitSim::FeatureEncoder::n_channels, BitSim::FeatureEncoder::observation_length} }) = observations->get(obs_time_idx);
        }
    }

    //timer.print_elapsed("DataLoader generate data");

    batch.past_observations = batch.past_observations.cuda();
    batch.future_positives = batch.future_positives.cuda();
    batch.future_negatives = batch.future_negatives.cuda();

    //Utils::save_tensor(batch.past_observations, "past_observations.tensor");

    return batch;
}

c10::optional<size_t> TradeDataset::size(void) const
{
    return BitSim::n_batches * BitSim::batch_size;
}
