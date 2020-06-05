#pragma once
#include "pch.h"

#include <thread>
#include <string>

#include "BitmexAccount.h"
#include "BitmexAuthentication.h"


class BitmexWebSocket : public std::enable_shared_from_this<BitmexWebSocket>
{
public:
    BitmexWebSocket(sptrBitmexAccount bitmex_account);

    void start(void);
    void shutdown(void);

private:
    sptrBitmexAccount bitmex_account;
    bool connected;

    boost::asio::io_context ioc;
    std::unique_ptr<boost::asio::ssl::context> ctx;
    std::unique_ptr<boost::asio::ip::tcp::resolver> resolver;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>> websocket;

    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    std::string host_address;
    boost::beast::flat_buffer websocket_buffer;

    BitmexAuthentication authenticator;

    double wallet;

    void connect(void);
    void request_authentication(void);
    void send(const std::string& message);
    void parse_message(const std::string& message);

    void fail(boost::beast::error_code ec, const std::string &reason);
    void on_resolve(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type results);
    void on_connect(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type::endpoint_type ep);
    void on_ssl_handshake(boost::beast::error_code ec);
    void on_handshake(boost::beast::error_code ec);
    void on_write(boost::beast::error_code ec, std::size_t bytes_transferred);
    void on_read(boost::beast::error_code ec, std::size_t bytes_transferred);
    void on_close(boost::beast::error_code ec);

    void websocket_worker(void);
};

using sptrBitmexWebSocket = std::shared_ptr<BitmexWebSocket>;
