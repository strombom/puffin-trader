#include "BitmexInterval.h"


BitmexInterval::BitmexInterval(sptrDatabase database) :
    database(database),
    interval_data_thread_running(true)
{
    interval_data_worker_thread = std::make_unique<std::thread>(&BitmexInterval::interval_data_worker, this);
}

void BitmexInterval::shutdown(void)
{
    interval_data_thread_running = false;
    interval_data_condition.notify_all();

    try {
        interval_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexInterval::update(void)
{

}

void BitmexInterval::interval_data_worker(void)
{
    while (interval_data_thread_running) {
        {
            auto interval_data_lock = std::unique_lock<std::mutex>{ interval_data_mutex };
            interval_data_condition.wait(interval_data_lock);
            if (!interval_data_thread_running) {
                break;
            }
        }

    }
}
