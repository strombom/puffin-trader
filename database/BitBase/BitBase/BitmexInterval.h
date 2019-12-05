#pragma once

#include "Database.h"
#include "DateTime.h"

#include <mutex>
#include <array>
#include <atomic>
#include <memory>
#include <thread>

using namespace std::chrono_literals;
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

    static const auto max_ticks_per_werk = 100;
    static constexpr auto intervals = std::array<std::chrono::seconds, 1>{120s};
};

using uptrBitmexInterval = std::unique_ptr<BitmexInterval>;
