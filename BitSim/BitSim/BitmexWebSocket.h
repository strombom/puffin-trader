#pragma once
#include "pch.h"

#include <thread>


class BitmexWebSocket
{
public:
    BitmexWebSocket(void);

    void start(void);
    void shutdown(void);

private:
    bool connected;
    boost::asio::io_context ioc;
    std::unique_ptr<boost::asio::ssl::context> ctx;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>>> websocket;

    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    void connect(void);
    void websocket_worker(void);
};

