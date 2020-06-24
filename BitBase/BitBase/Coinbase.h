#pragma once
#include "pch.h"

#include <stdio.h>
#include <mutex>
#include <thread>
#include <future>

#include "Database.h"
#include "CoinbaseTick.h"
#include "CoinbaseLive.h"
#include "CoinbaseInterval.h"


enum class CoinbaseState {
    idle,
    downloading_tick,
    downloading_live,
    shutdown
};

class Coinbase
{
public:
    Coinbase(sptrDatabase database);

    void shutdown(void);
    void tick_data_updated_callback(void);

private:
    sptrDatabase database;
    uptrCoinbaseTick Coinbase_tick;
    uptrCoinbaseLive Coinbase_live;
    uptrCoinbaseInterval Coinbase_interval;

    std::mutex state_mutex;
    bool thread_running;
    CoinbaseState state;

    std::mutex interval_update_mutex;
    std::condition_variable interval_update_condition;

    std::unique_ptr<std::thread> main_loop_thread;
    std::unique_ptr<std::thread> interval_update_thread;

    void main_loop(void);
    void interval_update_worker(void);
};
