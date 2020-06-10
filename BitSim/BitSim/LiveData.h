#pragma once

#include "DateTime.h"
#include "Intervals.h"
#include "BitBaseClient.h"
#include "FE_Inference.h"
#include "FE_Observations.h"

#include <thread>


class LiveData
{
public:
    LiveData(void);

    void start(void);
    void shutdown(void);

    std::tuple<bool, time_point_ms, torch::Tensor> get_next_interval(std::chrono::milliseconds timeout);

private:
    BitBaseClient bitbase_client;

    time_point_ms latest_timestamp;

    sptrIntervals intervals;
    sptrFE_Observations observations;
    torch::Tensor features;
    sptrFE_Inference feature_encoder;
    std::mutex new_data_mutex;
    std::condition_variable new_data_condition;

    std::atomic_bool live_data_thread_running;
    std::unique_ptr<std::thread> live_data_thread;

    void live_data_worker(void);
};

using sptrLiveData = std::shared_ptr<LiveData>;
