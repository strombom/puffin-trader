#pragma once
#include "pch.h"

#include "DateTime.h"
#include "Database.h"
#include "BitBotConstants.h"

#include <mutex>
#include <queue>
#include <string>


using tick_data_updated_callback_t = std::function<void(void)>;

enum class BinanceLiveState {
    idle,
    downloading
};

class BinanceLive
{
public:
    BinanceLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback);

    BinanceLiveState get_state(void);
    void start(void);
    void shutdown(void);

private:
    using TickData = std::map<std::string, uptrDatabaseTicks>;
    using uptrTickData = std::unique_ptr<TickData>;

    std::atomic<BinanceLiveState> state;

    sptrDatabase database;

    std::mutex tick_data_mutex;
    std::unique_ptr<std::thread> tick_data_worker_thread;
    std::deque<uptrTickData> tick_data_queue;
    std::condition_variable tick_data_condition;
    std::atomic_bool tick_data_thread_running;
    tick_data_updated_callback_t tick_data_updated_callback;

    zmq::context_t zmq_context;
    std::unique_ptr<zmq::socket_t> zmq_client;

    void tick_data_worker(void);
};

using uptrBinanceLive = std::unique_ptr<BinanceLive>;
