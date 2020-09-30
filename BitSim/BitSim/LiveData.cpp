#include "pch.h"

#include "BitLib/Logger.h"
#include "LiveData.h"


LiveData::LiveData(time_point_ms start_time) :
    live_data_thread_running(true),
    next_timestamp(start_time),
    agg_tick_idx(0)
{

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

sptrAggTick LiveData::get_next_agg_tick(void)
{
    auto lock = std::scoped_lock{ agg_ticks_mutex };
    if (agg_ticks.agg_ticks.size() > agg_tick_idx) {
        // TODO: If agg_tick_idx, remove elements from beginning of agg_ticks
        return std::make_shared<AggTick>(agg_ticks.agg_ticks[agg_tick_idx++]);
    }
    return nullptr;
}

void LiveData::live_data_worker(void)
{
    while (live_data_thread_running) {
        const auto new_ticks = bitbase_client.get_ticks(BitSim::symbol, BitSim::exchange, next_timestamp);

        for (auto& tick : new_ticks->rows) {
            auto lock = std::scoped_lock{ agg_ticks_mutex };
            agg_ticks.insert(tick);
            next_timestamp = tick.timestamp + 1ms;
        }
    }
}
