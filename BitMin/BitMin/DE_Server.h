#pragma once
#include "pch.h"

#include <atomic>
#include <thread>


class DE_Server
{
public:
    DE_Server(void);
    ~DE_Server(void);

private:
    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    void server_thread(void);
};
