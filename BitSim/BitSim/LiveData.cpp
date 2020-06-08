#include "pch.h"
#include "LiveData.h"


LiveData::LiveData(void) :
    live_data_thread_running(true)
{

}

void LiveData::start(void)
{
    live_data_thread = std::make_unique<std::thread>(&LiveData::live_data_worker, this);
}

void LiveData::shutdown(void)
{
    std::cout << "LiveData: Shutting down" << std::endl;
    live_data_thread_running = false;

    try {
        live_data_thread->join();
    }
    catch (...) {}
}

void LiveData::live_data_worker(void)
{
    while (live_data_thread_running) {
        std::this_thread::sleep_for(100ms);
    }
}
