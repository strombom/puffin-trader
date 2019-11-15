
#include "DownloadManager.h"
#include "Logger.h"

#include <future>
#include "curl/curl.h"


DownloadManager::DownloadManager(void)
{
    curl_global_init(CURL_GLOBAL_ALL);
}

DownloadManager::~DownloadManager(void)
{
    curl_global_cleanup();
}

std::shared_ptr<DownloadManager> DownloadManager::create(void)
{
    return std::make_shared<DownloadManager>();
}

void DownloadManager::download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done)
{
    std::scoped_lock lock(threads_mutex);

    threads.push_back(uptrDownloadThread(new DownloadThread(url, client_id, callback_arg, client_callback_done, std::bind(&DownloadManager::download_done_callback, this, _1, _2))));
    //logger.info("DownloadManager download %d", threads.size());
    start_next();
}

void DownloadManager::abort(std::string client_id)
{
    //logger.info("DownloadManager ABORT %s", client_id.c_str());
    std::scoped_lock lock(threads_mutex);

    for (auto&& thread : threads) {
        if (thread->test_id(client_id)) {
            thread->shutdown();
        }
    }

    for (auto&& thread = threads.begin(); thread != threads.end();) {
        if ((*thread)->test_id(client_id)) {
            (*thread)->join();
            thread = threads.erase(thread);
        }
        else {
            ++thread;
        }
    }
}

void DownloadManager::shutdown(void)
{
    std::scoped_lock lock(threads_mutex);

    for (auto&& thread : threads) {
        thread->shutdown();
    }
}

void DownloadManager::join(void)
{
    std::scoped_lock lock(threads_mutex);

    for (auto&& thread : threads) {
        thread->join();
    }
}

void DownloadManager::download_done_callback(std::string client_id, std::string callback_arg)
{
    std::scoped_lock lock(threads_mutex);

    active_threads_count--;

    //logger.info("DownloadManager callback count %d", threads.size());

    for (auto&& thread = threads.begin(); thread != threads.end(); ++thread) {
        //logger.info("DownloadManager compare id %s, %s", client_id.c_str(), callback_arg.c_str());

        if ((*thread)->test_id(client_id, callback_arg)) {
            //logger.info("Download done (%s) (%s)", client_id.c_str(), callback_arg.c_str());

           // (*thread)->join();
            //threads.erase(thread);
            break;
        }
    }
    //logger.info("DownloadManager callback done %s, %s", client_id.c_str(), callback_arg.c_str());

    start_next();
    /*
    logger.info("DownloadManager callback a %s, %s", client_id.c_str(), callback_arg.c_str());
    auto task = std::async(std::launch::async, &DownloadManager::download_done_callback_task, this, client_id, callback_arg, payload);
    logger.info("DownloadManager callback b %s, %s", client_id.c_str(), callback_arg.c_str());
    */
}

/*
void DownloadManager::download_done_callback_task(std::string client_id, std::string callback_arg, sptr_payload_t payload)
{
    std::scoped_lock lock(threads_mutex);

    active_threads_count--;

    logger.info("DownloadManager callback count %d", threads.size());

    for (auto&& thread = threads.begin(); thread != threads.end(); ++thread) {
        logger.info("DownloadManager compare id %s, %s", client_id.c_str(), callback_arg.c_str());

        if ((*thread)->test_id(client_id, callback_arg)) {
            logger.info("Download done (%s) (%s)", client_id.c_str(), callback_arg.c_str());
            (*thread)->join();
            threads.erase(thread);
            break;
        }
    }
    logger.info("DownloadManager callback done %s, %s", client_id.c_str(), callback_arg.c_str());

    start_next();
}
    */

void DownloadManager::start_next(void)
{
    if (active_threads_count == max_active_threads_count) {
        return;
    }

    for (auto&& thread = threads.begin(); thread != threads.end();) {
        if ((*thread)->get_state() == DownloadState::finished) {
            //(*thread)->join();
            thread = threads.erase(thread);
        }
        else {
            ++thread;
        }
    }

    for (auto&& thread : threads) {
        if (thread->get_state() == DownloadState::waiting_for_start) {
            thread->start();
            active_threads_count++;
            return;
        }
    }
}

//{
    /*
    while (threads.size() > 0 && threads[0]->get_state() == DownloadState::success)
    {
        printf("\nDownloaded (%s)\n", threads[0]->get_url().c_str());
        threads.pop_front();

        if (download_url_queue.size() > 0) {
            if (start_download(download_url_queue.front())) {
                download_url_queue.pop();
            }
        }
    }

    for (int idx = 0; idx < threads.size(); idx++) {
        if (threads[idx]->get_state() == DownloadState::failed) {
            threads[idx]->restart_download();
        }
    }
    */
//}

/*
void DownloadManager::download_progress_callback(void)
{
    printf("\33[2K\r Progress");
    for (int idx = 0; idx < threads.size(); idx++) {

        if (threads[idx]->get_state() == DownloadState::downloading) {
            printf("  % 5.1f MB ", threads[idx]->get_progress());

        } else if (threads[idx]->get_state() == DownloadState::success) {
            printf("  done ");

        } else if (threads[idx]->get_state() == DownloadState::failed) {
            printf("  failed ");
        }
    }
    fflush(stdout);
}
*/
