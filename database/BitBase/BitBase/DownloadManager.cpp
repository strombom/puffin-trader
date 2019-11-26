
#include "DownloadManager.h"
#include "Logger.h"

#include <future>
#include "curl/curl.h"


DownloadManager::DownloadManager(void)
{
    curl_global_init(CURL_GLOBAL_ALL);

    threads.reserve(threads_count);
    for (int i = 0; i < threads_count; ++i) {
        threads.push_back(sptrDownloadThread(new DownloadThread(std::bind(&DownloadManager::download_done_callback, this, std::placeholders::_1))));
    }
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
    std::scoped_lock lock(download_done_mutex, client_args_mutex);

    pending_tasks.clear();

    for (auto&& thread : threads) {
        thread->shutdown();
    }
}

void DownloadManager::join(void)
{
    for (auto&& thread : threads) {
        thread->join();
    }

    threads.clear();
}

void DownloadManager::abort_client(std::string client_id)
{
    {
        std::scoped_lock lock(download_done_mutex, client_args_mutex);
        for (auto&& task = pending_tasks.begin(); task != pending_tasks.end();) {
            if ((*task)->get_client_id() == client_id) {
                task = pending_tasks.erase(task);
            }
            else {
                ++task;
            }
        }

        for (auto&& thread : threads) {
            if (thread->test_client_id(client_id)) {
                thread->abort_download();
            }
        }
    }

    {
        std::scoped_lock lock(download_done_mutex, client_args_mutex);
        finished_tasks.erase(client_id);
    }    
}

void DownloadManager::download(std::string url, std::string client_id, client_callback_done_t client_callback_done)
{
    std::scoped_lock lock(client_args_mutex);

    if (next_download_id.find(client_id) == next_download_id.end()) {
        next_download_id[client_id] = 0;
        expected_download_id[client_id] = 0;
    }
    else {
        ++next_download_id[client_id];
    }

    uptrDownloadTask task = DownloadTask::create(url, client_id, next_download_id[client_id], client_callback_done);
    pending_tasks.push_back(std::move(task));

    while (start_pending_downloads());
}

void DownloadManager::download_done_callback(uptrDownloadTask task)
{
    std::scoped_lock lock(download_done_mutex);

    const std::string client_id = task->get_client_id();

    //logger.info("DownloadManager::download_done_callback (%s)", task->get_client_arg().c_str());
    finished_tasks[client_id].push_back(std::move(task));
    
    for (auto&& task = finished_tasks[client_id].begin(); task != finished_tasks[client_id].end();) {
        std::scoped_lock lock(client_args_mutex);
        if ((*task)->get_download_id() == expected_download_id[client_id]) {
            (*task)->run_client_callback();
            finished_tasks[client_id].erase(task);
            task = finished_tasks[client_id].begin();
            ++expected_download_id[client_id];
        }
        else {
            ++task;
        }
    }

    while (start_pending_downloads());
}

bool DownloadManager::start_pending_downloads(void)
{
    // Assign pending tasks to idle threads
    std::shared_ptr<DownloadThread> idle_thread;

    for (auto&& thread : threads) {
        if (thread->is_idle()) {
            idle_thread = thread;
            break;
        }
    }
    
    if (idle_thread) {
        idle_thread->assign_task(std::move(pending_tasks.front()));
        pending_tasks.pop_front();
        return true;
    }
    return false;
}
