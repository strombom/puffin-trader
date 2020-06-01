#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <string>

#include "TickData.h"


class BitmexWebSocket
{
public:
    BitmexWebSocket(sptrTickData tick_data);

    void start(void);
    void shutdown(void);

private:
    const char* host = "www.bitmex.com";
    const char* port = "443";
    const char* url = "/realtime";

    sptrTickData tick_data;

    boost::asio::io_context ioc;
    std::unique_ptr<boost::asio::ssl::context> ctx;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>>> websocket;

    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    void websocket_worker(void);
};
