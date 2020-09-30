#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"
#include "BitLib/AggTicks.h"
#include "BitLib/BitBaseClient.h"

#include <thread>


class LiveData
{
public:
    LiveData(time_point_ms start_time);

    void start(void);
    void shutdown(void);

    sptrAggTick get_next_agg_tick(void);

private:
    BitBaseClient bitbase_client;

    time_point_ms next_timestamp;
    std::mutex agg_ticks_mutex;
    AggTicks agg_ticks;
    size_t agg_tick_idx;

    std::atomic_bool live_data_thread_running;
    std::unique_ptr<std::thread> live_data_thread;

    void live_data_worker(void);
};

using sptrLiveData = std::shared_ptr<LiveData>;
