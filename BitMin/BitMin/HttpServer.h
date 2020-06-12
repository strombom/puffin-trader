#pragma once
#include "pch.h"

#include <thread>
#include <atomic>


class HttpServer
{
public:
    HttpServer(void);

    void start(void);
    void shutdown(void);

private:
    std::atomic_bool server_thread_running;
    std::unique_ptr<std::thread> server_thread;

    boost::asio::io_context ioc;

    void server_worker(void);
};
