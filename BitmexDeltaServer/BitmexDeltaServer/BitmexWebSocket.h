#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

/*
#include <cpprest/ws_client.h>
#include <cpprest/json.h>
*/
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
    const char* host = "www.bitmex.com";
    const char* url = "/realtime";

    sptrTickData tick_data;


    //boost::asio::io_context ioc;
    //boost::asio::ssl::context ctx;
    //boost::beast::websocket::stream<boost::asio::ip::tcp::socket> websocket;

    //std::unique_ptr<web::websockets::client::websocket_client> client;
    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    //bool json_test_field(const web::json::value& data, const std::string& name, const std::string& value);
    void websocket_worker(void);
};
