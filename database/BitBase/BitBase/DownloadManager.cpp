
#include "DownloadManager.h"
#include "Logger.h"

#include <future>
#include "curl/curl.h"


DownloadManager::DownloadManager(void)
{
    curl_global_init(CURL_GLOBAL_ALL);

    threads.reserve(threads_count);
    for (int i = 0; i < threads_count; ++i) {
        threads.emplace_back();
    }
}

DownloadManager::~DownloadManager(void)
{
    /*
    for (auto&& client : threads) {
        for (auto&& thread : client.second) {
            thread->shutdown();
        }
    }

    threads.clear();
    */
    curl_global_cleanup();
}

std::shared_ptr<DownloadManager> DownloadManager::create(void)
{
    return std::make_shared<DownloadManager>();
}

void DownloadManager::download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done)
{
    /*
    {
        std::scoped_lock lock(threads_mutex);
        threads[client_id].push_back(uptrDownloadThread(new DownloadThread(url, client_id, callback_arg, client_callback_done, std::bind(&DownloadManager::download_done_callback, this, _1, _2))));
    }

    manage_threads();*/
}

void DownloadManager::abort(std::string client_id)
{
    /*
    for (auto&& client : threads) {
        for (auto&& thread : client.second) {
            if (thread->test_id(client_id)) {
                thread->shutdown();
            }
        }
    }

    {
        std::scoped_lock lock(threads_mutex);
        for (auto&& client : threads) {
            for (auto&& thread = client.second.begin(); thread != client.second.end();) {
                if ((*thread)->test_id(client_id)) {
                    thread = client.second.erase(thread);
                }
                else {
                    ++thread;
                }
            }
        }
    }*/
}

void DownloadManager::shutdown(void)
{
    /*
    for (auto&& client : threads) {
        for (auto&& thread : client.second) {
            thread->shutdown();
        }
    }
    */
}

void DownloadManager::join(void)
{
    /*
    for (auto&& client : threads) {
        for (auto&& thread : client.second) {
            thread->join();
        }
    }
    */
}

/*
void DownloadManager::download_done_callback(std::string client_id, std::string callback_arg)
{
    logger.info("DownloadManager::download_done_callback (%s, %s) count %d", client_id.c_str(), callback_arg.c_str(), threads.size());

    manage_threads();
}
*/

/*
void DownloadManager::manage_threads(void)
{
    logger.info("DownloadManager::start_next");

    {
        std::scoped_lock lock(threads_mutex);

        for (auto&& client : threads) {
            for (auto&& thread = client.second.begin(); thread != client.second.end();) {
                if ((*thread)->has_data()) {
                    (*thread)->pass_data_to_client();
                    thread = client.second.erase(thread);
                } else {
                    break;
                }
            }

        }
    }
    
    for (auto&& client : threads) {
        for (auto&& thread : client.second) {

        }
    }
}
*/

    /*
    for (auto&& thread : threads) {
        if (active_threads_count == max_active_threads_count) {
            return;
        }
        if (thread->is_ready_to_start()) {
            thread->start();
            active_threads_count++;
        }
    }
    */
