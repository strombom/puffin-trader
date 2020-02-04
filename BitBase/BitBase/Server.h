#pragma once
#include "pch.h"

#include "Database.h"

#include <atomic>
#include <thread>


class Server
{
public:
    Server(sptrDatabase database);
    ~Server(void);

private:
    sptrDatabase database;

    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    void server_thread(void);
};
