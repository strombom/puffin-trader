#pragma once

#include "CoinbaseProRestApi.h"
#include "TickData.h"

#include <thread>


class CoinbaseProTick
{
public:
    CoinbaseProTick(sptrTickData tick_data);

    void start(void);
    void shutdown(void);

private:
    sptrTickData tick_data;
    sptrCoinbaseProRestApi rest_api;

    std::unordered_map<std::string, long long> last_ids;

    std::atomic_bool tick_thread_running;
    std::unique_ptr<std::thread> tick_thread;

    void tick_worker(void);
};
