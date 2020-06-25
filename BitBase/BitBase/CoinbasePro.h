#pragma once
#include "pch.h"

#include <stdio.h>
#include <mutex>
#include <thread>
#include <future>

#include "Database.h"
#include "CoinbaseProTick.h"
#include "CoinbaseProLive.h"
#include "CoinbaseProInterval.h"


enum class CoinbaseProState {
    idle,
    downloading_tick,
    downloading_live,
    shutdown
};

class CoinbasePro
{
public:
    CoinbasePro(sptrDatabase database);

    void shutdown(void);
    void tick_data_updated_callback(void);

private:
    sptrDatabase database;
    uptrCoinbaseProTick CoinbasePro_tick;
    uptrCoinbaseProLive CoinbasePro_live;
    uptrCoinbaseProInterval CoinbasePro_interval;

    std::mutex state_mutex;
    bool thread_running;
    CoinbaseProState state;

    std::mutex interval_update_mutex;
    std::condition_variable interval_update_condition;

    std::unique_ptr<std::thread> main_loop_thread;
    std::unique_ptr<std::thread> interval_update_thread;

    void main_loop(void);
    void interval_update_worker(void);
};
