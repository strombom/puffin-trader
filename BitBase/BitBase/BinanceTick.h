#pragma once
#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/DateTime.h"
#include "BinanceRestApi.h"
#include "Database.h"

#include <mutex>
#include <queue>
#include <string>


using tick_data_updated_callback_t = std::function<void(void)>;

enum class BinanceTickState {
    idle,
    downloading
};

class BinanceTick
{
public:
    BinanceTick(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback);

    BinanceTickState get_state(void);
    void start(void);
    void shutdown(void);

private:
    sptrBinanceRestApi rest_api;
    sptrDatabase database;

    using TickData = std::map<std::string, uptrDatabaseTicks>;
    using uptrTickData = std::unique_ptr<TickData>;

    std::atomic<BinanceTickState> state;


    std::mutex tick_data_mutex;
    std::unique_ptr<std::thread> tick_data_worker_thread;
    std::deque<uptrTickData> tick_data_queue;
    std::condition_variable tick_data_condition;
    std::atomic_bool tick_data_thread_running;
    tick_data_updated_callback_t tick_data_updated_callback;

    zmq::context_t zmq_context;
    std::unique_ptr<zmq::socket_t> zmq_client;

    void insert_symbol_name(const std::string& new_symbol_name);
    void tick_data_worker(void);
};

using uptrBinanceTick = std::unique_ptr<BinanceTick>;
