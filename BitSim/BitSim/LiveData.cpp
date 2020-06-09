#include "pch.h"

#include "Logger.h"
#include "LiveData.h"


LiveData::LiveData(void) :
    live_data_thread_running(true)
{
    feature_encoder = std::make_shared<FE_Inference>(BitSim::tmp_path, BitSim::feature_encoder_weights_filename);

    const auto timestamp_now = system_clock_ms_now();
    const auto timestamp_start = timestamp_now - BitSim::LiveData::intervals_buffer_length - (timestamp_now - BitSim::timestamp_start) % BitSim::BitBase::interval;
    intervals = bitbase_client.get_intervals(BitSim::symbol, BitSim::exchange, timestamp_start, BitSim::BitBase::interval);
    observations = std::make_shared<FE_Observations>(intervals);
    features = feature_encoder->forward(observations->get_all());

    logger.info("LiveData::LiveData: Intervals (%d): %s", intervals->rows.size(), DateTime::to_string_iso_8601(intervals->timestamp_start).c_str());
    logger.info("LiveData::LiveData: Observations (%d)", observations->size());
}

void LiveData::start(void)
{
    live_data_thread = std::make_unique<std::thread>(&LiveData::live_data_worker, this);
}

void LiveData::shutdown(void)
{
    std::cout << "LiveData: Shutting down" << std::endl;
    live_data_thread_running = false;

    try {
        live_data_thread->join();
    }
    catch (...) {}
}

void LiveData::live_data_worker(void)
{
    while (live_data_thread_running) {
        std::this_thread::sleep_for(100ms);

        const auto timestamp_next = intervals->get_timestamp_last() + BitSim::BitBase::interval;
        const auto new_intervals = bitbase_client.get_intervals(BitSim::symbol, BitSim::exchange, timestamp_next, BitSim::BitBase::interval);

        if (new_intervals->rows.size() > 0) {
            intervals->rotate_insert(new_intervals);
            observations->rotate_insert(intervals, new_intervals->rows.size());

            // Encode new features
            const auto new_observations_size = std::min(new_intervals->rows.size(), observations->size());
            const auto new_observations = observations->get_tail((int)new_observations_size);
            const auto new_features = feature_encoder->forward(new_observations);
            if (new_features.size(0) < features.size(0)) {
                features.roll(-new_features.size(0), 0);
            }
            features.slice(0, features.size(0) - new_features.size(0)) = new_features;
            logger.info("LiveData::live_data_worker: New features %s int(%d) obs(%d) feat(%d)", DateTime::to_string_iso_8601(new_intervals->timestamp_start).c_str(), new_intervals->rows.size(), new_observations_size, new_features.size(0));
        }
    }
}
