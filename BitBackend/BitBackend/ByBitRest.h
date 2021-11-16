#pragma once
#include "precompiled_headers.h"
#include <nghttp2/asio_http2_client.h>

#include "BitLib/DateTime.h"
#include "ByBitAuthentication.h"


class ByBitRest
{
public:
    ByBitRest(void);

    int place_order(const std::string& symbol, double qty, double price);
    void cancel_order(const std::string& symbol, int _user_order_id);

    void join(void);
    bool is_connected(void);

private:
    int user_order_id;

    ByBitAuthentication authenticator;

    const std::string host = "api.bybit.com";
    const std::string service = "https";

    std::unique_ptr<boost::asio::ssl::context> tls_ctx;
    std::unique_ptr<boost::asio::io_service> io_service;
    std::unique_ptr<nghttp2::asio_http2::client::session> session;
    bool connected;
    
    void on_data(const char* data, std::size_t len);
    void get_request(const std::string& uri);
    void post_request(const std::string& uri, const std::string& data);
    void heartbeat_reset(void);

    void http2_runner(void);
    void heartbeat_runner(void);

    std::atomic_bool http2_thread_running;
    std::unique_ptr<std::thread> http2_thread;
    std::atomic_bool heartbeat_thread_running;
    std::unique_ptr<std::thread> heartbeat_thread;

    time_point_us heartbeat_timeout;
};
