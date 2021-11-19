#pragma once
#include "precompiled_headers.h"

#include "ByBitAuthentication.h"
#include "BitLib/DateTime.h"
#include "OrderManager.h"
#include "ByBitConfig.h"
#include "Symbols.h"

#include <nghttp2/asio_http2_client.h>


class ByBitRest
{
public:
    ByBitRest(sptrOrderManager order_manager);

    void place_order(const Symbol& symbol, double qty, double price);
    void cancel_all_orders(const Symbol& symbol);
    void cancel_order(const Symbol& symbol, Uuid id_external);
    void get_position(const Symbol& symbol);

    void join(void);
    bool is_connected(void);

private:
    sptrOrderManager order_manager;
    ByBitAuthentication authenticator;
    simdjson::ondemand::parser json_parser;

    std::unique_ptr<boost::asio::ssl::context> tls_ctx;
    std::unique_ptr<boost::asio::io_service> io_service;
    std::unique_ptr<nghttp2::asio_http2::client::session> session;
    bool connected;
    
    void get_request(const std::string& query, ByBit::Rest::Endpoint endpoint);
    void post_request(const std::string& data, ByBit::Rest::Endpoint endpoint);
    void on_data(const char* data, std::size_t len, ByBit::Rest::Endpoint endpoint);
    void heartbeat_reset(void);

    void http2_runner(void);
    void heartbeat_runner(void);

    std::atomic_bool http2_thread_running;
    std::unique_ptr<std::thread> http2_thread;
    std::atomic_bool heartbeat_thread_running;
    std::unique_ptr<std::thread> heartbeat_thread;

    time_point_us heartbeat_timeout;
};
