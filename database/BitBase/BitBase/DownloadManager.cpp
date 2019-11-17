
#include "DownloadManager.h"
#include "Logger.h"

#include <future>
#include "curl/curl.h"


DownloadManager::DownloadManager(void)
{
    curl_global_init(CURL_GLOBAL_ALL);

    threads.reserve(threads_count);
    for (int i = 0; i < threads_count; ++i) {
        threads.push_back(sptrDownloadThread(new DownloadThread(std::bind(&DownloadManager::download_done_callback, this, _1))));
    }
}

DownloadManager::~DownloadManager(void)
{
    shutdown();
    join();
    curl_global_cleanup();
}

void DownloadManager::shutdown(void)
{
    for (auto&& thread : threads) {
        thread->shutdown();
    }
}

void DownloadManager::join(void)
{
    for (auto&& thread : threads) {
        thread->join();
    }
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
                thread->abort();
            }
        }
    }

    {
        std::scoped_lock lock(download_done_mutex, client_args_mutex);
        for (auto&& task = finished_tasks[client_id].begin(); task != finished_tasks[client_id].end();) {
            if ((*task)->get_client_id() == client_id) {
                task = pending_tasks.erase(task);
            }
            else {
                ++task;
            }
        }
    }
    
}

std::shared_ptr<DownloadManager> DownloadManager::create(void)
{
    return std::make_shared<DownloadManager>();
}

void DownloadManager::download(std::string url, std::string client_id, std::string client_arg, client_callback_done_t client_callback_done)
{
    std::scoped_lock lock(client_args_mutex);

    //logger.info("DownloadManager::download %s", client_arg.c_str());
    uptrDownloadTask task = DownloadTask::create(url, client_id, client_arg, client_callback_done);
    pending_tasks.push_back(std::move(task));
    client_args[client_id].push(client_arg);

    work();
}

void DownloadManager::download_done_callback(uptrDownloadTask task)
{
    std::scoped_lock lock(download_done_mutex);

    const std::string client_id = task->get_client_id();

    //logger.info("DownloadManager::download_done_callback (%s)", task->get_client_arg().c_str());
    finished_tasks[client_id].push_back(std::move(task));
    
    for (auto&& task = finished_tasks[client_id].begin(); task != finished_tasks[client_id].end();) {
        const std::string client_arg = client_args[client_id].front();
        if ((*task)->get_client_arg() == client_arg) {
            (*task)->run_client_callback();
            finished_tasks[client_id].erase(task);
            {
                std::scoped_lock lock(client_args_mutex);
                client_args[client_id].pop();
            }
            task = finished_tasks[client_id].begin();
        }
        else {
            ++task;
        }
    }

    work();
}

void DownloadManager::work(void)
{
    // Assign pending tasks to idle threads
    while (!pending_tasks.empty()) {
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
        } else {
            break;
        }
    }
}
