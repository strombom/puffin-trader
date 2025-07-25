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

#include "OrderBookData.h"


class BitmexWebSocket
{
public:
    BitmexWebSocket(sptrOrderBookData order_book_data);

    void start(void);
    void shutdown(void);

private:
    const char* host = "www.bitmex.com";
    const char* port = "443";
    const char* url = "/realtime";

    sptrOrderBookData order_book_data;

    bool connected;
    boost::asio::io_context ioc;
    std::unique_ptr<boost::asio::ssl::context> ctx;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>>> websocket;

    std::map<std::string, OrderBook> previous_order_book;

    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    void connect(void);
    void websocket_worker(void);
};
