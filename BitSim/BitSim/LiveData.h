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
    std::mutex tick_data_mutex;
    std::unique_ptr<std::thread> tick_data_worker_thread;
    std::condition_variable tick_data_condition;
    std::atomic_bool tick_data_thread_running;

    zmq::context_t zmq_context;
    std::unique_ptr<zmq::socket_t> zmq_client;

    time_point_ms next_timestamp;
    std::mutex agg_ticks_mutex;
    AggTicks agg_ticks;
    size_t agg_tick_idx;

    void live_data_connect(void);
    void tick_data_worker(void);

    //BitBaseClient bitbase_client;
    //std::atomic_bool live_data_thread_running;
    //std::unique_ptr<std::thread> live_data_thread;
    //void live_data_worker(void);
};

using sptrLiveData = std::shared_ptr<LiveData>;
