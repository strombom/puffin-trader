#pragma once

#include <cpprest/ws_client.h>

#include <thread>


class BitmexWebSocket
{
public:
    BitmexWebSocket(void);

    void start(void);
    void shutdown(void);

private:
    const char* ws_url = "wss://www.bitmex.com/realtime";

    std::unique_ptr<web::websockets::client::websocket_client> client;
    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    void websocket_worker(void);
};
