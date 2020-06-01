#pragma once

#include <atomic>
#include <thread>

#include "TickData.h"


#include <mutex>
#include <condition_variable>
		

class Server
{
public:
    Server(sptrTickData tick_data);
    ~Server(void);

    void test(void);

private:
    std::mutex test_mutex;
    std::condition_variable test_condition;



    sptrTickData tick_data;

    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    void server_thread(void);
};
