#pragma once

#include "Database.h"
#include "DateTime.h"
#include "BitBaseConstants.h"

#include <mutex>
#include <atomic>
#include <memory>
#include <thread>


using uptrThread = std::unique_ptr<std::thread>;

class BitmexInterval
{
public:
    BitmexInterval(sptrDatabase database);

    void shutdown(void);

    void update(void);

private:
    sptrDatabase database;

    std::mutex interval_data_mutex;
    std::atomic_bool interval_data_thread_running;
    std::condition_variable interval_data_condition;
    uptrThread interval_data_worker_thread;

    void interval_data_worker(void);
};

using uptrBitmexInterval = std::unique_ptr<BitmexInterval>;
