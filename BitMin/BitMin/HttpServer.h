#pragma once
#include "pch.h"

#include <chrono>
#include <thread>
#include <atomic>


class HttpListener : public std::enable_shared_from_this<HttpListener>
{
public:
    HttpListener(std::shared_ptr<boost::asio::io_context> ioc, std::shared_ptr<boost::asio::ssl::context> ctx, boost::asio::ip::tcp::endpoint endpoint);

    std::shared_ptr<boost::asio::io_context> ioc;
    std::shared_ptr<boost::asio::ssl::context> ctx;
    boost::asio::ip::tcp::acceptor acceptor;


    void run(void);

private:
    void do_accept(void);
    void on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket);
};

class HttpServer
{
public:
    HttpServer(void);

    void start(void);
    void shutdown(void);

private:
    std::atomic_bool server_thread_running;
    std::unique_ptr<std::thread> server_thread;

    std::shared_ptr<boost::asio::io_context> ioc;
    std::shared_ptr<boost::asio::ssl::context> ctx;

    void server_worker(void);
};
