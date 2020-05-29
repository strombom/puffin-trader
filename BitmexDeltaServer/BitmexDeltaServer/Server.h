#pragma once

#include <atomic>
#include <thread>

#include "TickData.h"


class Server
{
public:
    Server(sptrTickData tick_data);
    ~Server(void);

private:
    sptrTickData tick_data;

    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    void server_thread(void);
};
