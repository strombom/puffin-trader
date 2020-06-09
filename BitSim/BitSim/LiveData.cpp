#include "pch.h"

#include "Logger.h"
#include "LiveData.h"


LiveData::LiveData(void) :
    live_data_thread_running(true)
{
    const auto timestamp_now = system_clock_ms_now();
    const auto timestamp_start = timestamp_now - BitSim::LiveData::intervals_buffer_length - (timestamp_now - BitSim::timestamp_start) % BitSim::BitBase::interval;
    intervals = bitbase_client.get_intervals(BitSim::symbol, BitSim::exchange, timestamp_start, BitSim::BitBase::interval);
    logger.info("LiveData::LiveData: Intervals (%d): %s", intervals->rows.size(), DateTime::to_string_iso_8601(intervals->timestamp_start).c_str());
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
        std::this_thread::sleep_for(250ms);

        const auto timestamp_next = intervals->get_timestamp_last() + BitSim::BitBase::interval;
        const auto new_intervals = bitbase_client.get_intervals(BitSim::symbol, BitSim::exchange, timestamp_next, BitSim::BitBase::interval);

        logger.info("LiveData::live_data_worker: New intervals (%d): %s", new_intervals->rows.size(), DateTime::to_string_iso_8601(new_intervals->timestamp_start).c_str());
    }
}
