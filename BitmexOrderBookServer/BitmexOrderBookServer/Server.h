#pragma once

#include <atomic>
#include <thread>

#include "OrderBookData.h"


#include <mutex>
#include <condition_variable>
		

class Server
{
public:
    Server(sptrOrderBookData order_book_data);
    ~Server(void);

    void test(void);

private:
    std::mutex test_mutex;
    std::condition_variable test_condition;

    sptrOrderBookData order_book_data;

    std::atomic_bool server_running;
    std::unique_ptr<std::thread> server_thread_handle;

    void server_thread(void);
};
