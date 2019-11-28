
#include "DownloadManager.h"
#include "Logger.h"

#include <future>
#include "curl/curl.h"


DownloadManager::DownloadManager(void)
{
    curl_global_init(CURL_GLOBAL_ALL);

    threads.reserve(threads_count);
    for (int i = 0; i < threads_count; ++i) {
        auto thread = std::make_shared<DownloadThread>(std::bind(&DownloadManager::download_done_callback, this, std::placeholders::_1));
        threads.push_back(thread);
    }

    pending_tasks_thread_handle = std::make_unique<std::thread>(&DownloadManager::pending_tasks_thread, this);
    finished_tasks_thread_handle = std::make_unique<std::thread>(&DownloadManager::finished_tasks_thread, this);
}

DownloadManager::~DownloadManager(void)
{
    shutdown();
    join();
    curl_global_cleanup();
}

std::shared_ptr<DownloadManager> DownloadManager::create(void)
{
    return std::make_shared<DownloadManager>();
}

void DownloadManager::shutdown(void)
{
    std::scoped_lock thread_lock(threads_mutex);

    running = false;

    for (auto&& thread : threads) {
        thread->shutdown();
    }
}

void DownloadManager::join(void)
{
    std::scoped_lock thread_lock(threads_mutex);

    for (auto&& thread : threads) {
        thread->join();
    }

    threads.clear();

    pending_tasks_thread_handle->join();
    finished_tasks_thread_handle->join();
}

void DownloadManager::abort_client(std::string client_id)
{
    logger.info("DownloadManager::abort_client pending tasks");
    {
        std::scoped_lock lock(pending_tasks_mutex);

        for (auto&& task = pending_tasks.begin(); task != pending_tasks.end();) {
            if ((*task)->get_client_id() == client_id) {
                task = pending_tasks.erase(task);
            }
            else {
                ++task;
            }
        }
    }

    logger.info("DownloadManager::abort_client threads");

    {
        std::scoped_lock lock(threads_mutex);

        for (auto&& thread : threads) {
            if (thread->test_client_id(client_id)) {
                thread->abort_download();
            }
        }
    }

    logger.info("DownloadManager::abort_client finished tasks");

    {
        std::scoped_lock lock(finished_tasks_mutex);

        for (auto&& task = finished_tasks.begin(); task != finished_tasks.end();) {
            if ((*task)->get_client_id() == client_id) {
                task = finished_tasks.erase(task);
            }
            else {
                ++task;
            }
        }
    }
    logger.info("DownloadManager::abort_client end");
}

void DownloadManager::download(std::string url, std::string client_id, client_callback_done_t client_callback_done)
{
    {
        std::scoped_lock lock(pending_tasks_mutex);

        if (next_download_id.find(client_id) == next_download_id.end()) {
            next_download_id[client_id] = 0;
            expected_download_id[client_id] = 0;
        }
        else {
            ++next_download_id[client_id];
        }

        uptrDownloadTask task = DownloadTask::create(url, client_id, next_download_id[client_id], client_callback_done);
        pending_tasks.push_back(std::move(task));

    }

    pending_tasks_condition.notify_one();
}

void DownloadManager::download_done_callback(uptrDownloadTask task)
{
    {
        std::scoped_lock lock(finished_tasks_mutex);

        finished_tasks.push_back(std::move(task));
    }

    finished_tasks_condition.notify_one();
}

void DownloadManager::pending_tasks_thread(void)
{
    while (running) {
        std::unique_lock<std::mutex> worker_lock(pending_tasks_mutex);
        pending_tasks_condition.wait(worker_lock);

        while (!pending_tasks.empty() && running) {
            std::scoped_lock thread_lock(threads_mutex);

            std::shared_ptr<DownloadThread> idle_thread;
            for (auto&& thread : threads) {
                if (thread->is_idle()) {
                    idle_thread = thread;
                    break;
                }
            }

            if (!idle_thread) {
                break;
            }

            idle_thread->assign_task(std::move(pending_tasks.front()));
            pending_tasks.pop_front();
        }
    }

    std::scoped_lock<std::mutex> lock(pending_tasks_mutex);
    pending_tasks.clear();
}

void DownloadManager::finished_tasks_thread(void)
{
    while (running) {
        std::unique_lock<std::mutex> worker_lock(finished_tasks_mutex);
        finished_tasks_condition.wait(worker_lock);

        for (auto&& task = finished_tasks.begin(); task != finished_tasks.end() && running;) {
            const auto client_id = (*task)->get_client_id();
            const auto download_id = (*task)->get_download_id();

            if (download_id == expected_download_id[client_id]) {
                (*task)->call_client_callback();
                ++expected_download_id[client_id];
                finished_tasks.erase(task);
                task = finished_tasks.begin();
            }
            else {
                ++task;
            }
        }
    }

    std::scoped_lock<std::mutex> lock(finished_tasks_mutex);
    finished_tasks.clear();
}
