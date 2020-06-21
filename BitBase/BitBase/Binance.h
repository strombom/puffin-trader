#pragma once
#include "pch.h"

#include <stdio.h>
#include <mutex>
#include <thread>
#include <future>

#include "Database.h"
#include "BinanceTick.h"
#include "BinanceLive.h"
#include "BinanceInterval.h"


enum class BinanceState {
    idle,
    downloading_tick,
    downloading_live,
    shutdown
};

class Binance
{
public:
    Binance(sptrDatabase database);

    void shutdown(void);
    void tick_data_updated_callback(void);

private:
    sptrDatabase database;
    uptrBinanceTick binance_tick;
    uptrBinanceLive binance_live;
    uptrBinanceInterval binance_interval;

    std::mutex state_mutex;
    bool thread_running;
    BinanceState state;

    std::mutex interval_update_mutex;
    std::condition_variable interval_update_condition;

    std::unique_ptr<std::thread> main_loop_thread;
    std::unique_ptr<std::thread> interval_update_thread;

    void main_loop(void);
    void interval_update_worker(void);
};
