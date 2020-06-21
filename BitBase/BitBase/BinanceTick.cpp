#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BinanceTick.h"
#include "BitLib/DateTime.h"
#include "BitLib/Logger.h"

#include <msgpack.hpp>

#include <array>
#include <regex>
#include <string>
#include <iostream>


BinanceTick::BinanceTick(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BinanceTickState::idle), tick_data_thread_running(true)
{
    //zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    //zmq_client->connect(BitBase::Binance::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&BinanceTick::tick_data_worker, this);
}

BinanceTickState BinanceTick::get_state(void)
{
    return state;
}

void BinanceTick::shutdown(void)
{
    logger.info("BinanceTick::shutdown");

    state = BinanceTickState::idle;

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BinanceTick::start(void)
{
    assert(state == BinanceTickState::idle);
    state = BinanceTickState::downloading;
    tick_data_condition.notify_one();
}

void BinanceTick::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }

        
        tick_data_updated_callback();
        state = BinanceTickState::idle;
    }
    logger.info("BinanceTick::tick_data_worker exit");
}
