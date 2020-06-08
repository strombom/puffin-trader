#pragma once

#include "BitBaseClient.h"

#include <thread>


class LiveData
{
public:
    LiveData(void);

    void start(void);
    void shutdown(void);

private:
    BitBaseClient bitbase_client;

    std::atomic_bool live_data_thread_running;
    std::unique_ptr<std::thread> live_data_thread;

    void live_data_worker(void);
};
