#include "pch.h"

#include "FE_DataLoader.h"
#include "Utils.h"
#include "DateTime.h"


Batch TradeDataset::get_batch(c10::ArrayRef<size_t> request)
{
    auto timer = Timer();

    const auto shape = c10::IntArrayRef{ BitSim::n_channels, BitSim::observation_length };
    const auto batch_size = (int)request.size();
    auto batch = Batch{ batch_size };

    for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto time_index = random_index.get();

        for (auto obs_idx = 0; obs_idx < BitSim::n_observations; ++obs_idx) {
            const auto obs_time_idx = time_index - (BitSim::n_observations - obs_idx) * BitSim::observation_length;
            batch.past_observations[batch_idx].slice(1, obs_idx, obs_idx + 1, 1).reshape(shape) = observations->get(obs_time_idx);
        }

        for (auto pred_idx = 0; pred_idx < BitSim::n_predictions * BitSim::n_positive; ++pred_idx) {
            const auto obs_time_idx = time_index + pred_idx * BitSim::observation_length;
            batch.future_positives[batch_idx].slice(1, pred_idx, pred_idx + 1, 1).reshape(shape) = observations->get(obs_time_idx);
        }

        for (auto neg_idx = 0; neg_idx < BitSim::n_predictions * BitSim::n_negative; ++neg_idx) {
            const auto obs_time_idx = random_index.get();
            batch.future_negatives[batch_idx].slice(1, neg_idx, neg_idx + 1, 1).reshape(shape) = observations->get(obs_time_idx);
        }
    }

    //timer.print_elapsed("DataLoader generate data");
    //timer.restart();

    batch.past_observations = batch.past_observations.cuda();
    batch.future_positives = batch.future_positives.cuda();
    batch.future_negatives = batch.future_negatives.cuda();

    //timer.print_elapsed("DataLoader move to cuda");
    //Utils::save_tensor(batch.past_observations, "past_observations.tensor");

    return batch;
}

c10::optional<size_t> TradeDataset::size(void) const
{
    return BitSim::n_batches * BitSim::batch_size;
}

/*
for (auto obs_idx = 0; obs_idx < BitSim::n_observations; ++obs_idx) {
    const auto first_time_idx = time_index - (BitSim::n_observations - obs_idx) * BitSim::observation_length;
    const auto first_price = intervals.rows[first_time_idx].last_price;

    for (auto interval_idx = 0; interval_idx < BitSim::observation_length; ++interval_idx) {
        const auto row = &intervals.rows[(int)((long)first_time_idx + interval_idx)];

        batch.past_observations[batch_idx][BitSim::ch_price][obs_idx][interval_idx] = price_transform(first_price, row->last_price);
        batch.past_observations[batch_idx][BitSim::ch_buy_volume][obs_idx][interval_idx] = volume_transform(row->vol_buy);
        batch.past_observations[batch_idx][BitSim::ch_sell_volume][obs_idx][interval_idx] = volume_transform(row->vol_sell);
    }
}

for (auto pred_idx = 0; pred_idx < BitSim::n_predictions * BitSim::n_positive; ++pred_idx) {
    const auto first_time_idx = time_index + pred_idx * BitSim::observation_length;
    const auto first_price = intervals.rows[first_time_idx].last_price;

    for (auto interval_idx = 0; interval_idx < BitSim::observation_length; ++interval_idx) {
        const auto row = &intervals.rows[(int)((long)first_time_idx + interval_idx)];

        batch.future_positives[batch_idx][BitSim::ch_price][pred_idx][interval_idx] = price_transform(first_price, row->last_price);
        batch.future_positives[batch_idx][BitSim::ch_buy_volume][pred_idx][interval_idx] = volume_transform(row->vol_buy);
        batch.future_positives[batch_idx][BitSim::ch_sell_volume][pred_idx][interval_idx] = volume_transform(row->vol_sell);
    }
}

for (auto pred_idx = 0; pred_idx < BitSim::n_predictions * BitSim::n_negative; ++pred_idx) {
    const auto negative_index = random_index.get();
    const auto first_price = intervals.rows[negative_index].last_price;

    for (auto interval_idx = 0; interval_idx < BitSim::observation_length; ++interval_idx) {
        const auto row = &intervals.rows[(int)((long)negative_index + interval_idx)];

        batch.future_negatives[batch_idx][BitSim::ch_price][pred_idx][interval_idx] = price_transform(first_price, row->last_price);
        batch.future_negatives[batch_idx][BitSim::ch_buy_volume][pred_idx][interval_idx] = volume_transform(row->vol_buy);
        batch.future_negatives[batch_idx][BitSim::ch_sell_volume][pred_idx][interval_idx] = volume_transform(row->vol_sell);
    }
}
*/