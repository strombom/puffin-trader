#pragma once
#include "pch.h"

#include "BitLib/json11/json11.hpp"

#include <atomic>
#include <thread>


class DE_Server
{
public:
    DE_Server(void);
    ~DE_Server(void);

    json11::Json get_direction_data(void);

private:
    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    json11::Json direction_data;

    void server_thread(void);
};
