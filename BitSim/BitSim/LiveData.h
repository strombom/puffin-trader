#pragma once

#include "DateTime.h"
#include "Intervals.h"
#include "BitBaseClient.h"
#include "FE_Observations.h"

#include <thread>


class LiveData
{
public:
    LiveData(void);

    void start(void);
    void shutdown(void);

private:
    BitBaseClient bitbase_client;

    sptrIntervals intervals;
    sptrFE_Observations observations;

    std::atomic_bool live_data_thread_running;
    std::unique_ptr<std::thread> live_data_thread;

    void live_data_worker(void);
};
