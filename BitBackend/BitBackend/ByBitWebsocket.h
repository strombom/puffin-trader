#pragma once

#include "ByBitAuthentication.h"

#include <boost/asio/io_context.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <thread>


class ByBitWebSocket : public std::enable_shared_from_this<ByBitWebSocket>
{
public:
    ByBitWebSocket(void);

    void start(void);
    void shutdown(void);

private:
    ByBitAuthentication authenticator;
    bool connected;

    boost::asio::io_context ioc;
    std::unique_ptr<boost::asio::ssl::context> ctx;
    std::unique_ptr<boost::asio::ip::tcp::resolver> resolver;
    std::unique_ptr<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>> websocket;

    std::atomic_bool websocket_thread_running;
    std::unique_ptr<std::thread> websocket_thread;

    std::string host_address;
    boost::beast::flat_buffer websocket_buffer;

    void connect(void);
    void request_authentication(void);
    void send(const std::string& message);
    void parse_message(const std::string& message);

    void fail(boost::beast::error_code ec, const std::string& reason);
    void on_resolve(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type results);
    void on_connect(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type::endpoint_type ep);
    void on_ssl_handshake(boost::beast::error_code ec);
    void on_handshake(boost::beast::error_code ec);
    void on_write(boost::beast::error_code ec, std::size_t bytes_transferred);
    void on_read(boost::beast::error_code ec, std::size_t bytes_transferred);
    void on_close(boost::beast::error_code ec);

    void websocket_worker(void);
};
