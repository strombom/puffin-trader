#pragma once

#include <cpprest/ws_client.h>
#include <cpprest/json.h>
#include "TickData.h"

#include <thread>


class BitmexWebSocket
{
public:
    BitmexWebSocket(sptrTickData tick_data);

    void start(void);
    void shutdown(void);

private:
    const char* ws_url = "wss://www.bitmex.com/realtime";

    sptrTickData tick_data;

    std::unique_ptr<web::websockets::client::websocket_client> client;
    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    bool json_test_field(const web::json::value& data, const std::string& name, const std::string& value);
    void websocket_worker(void);
};
