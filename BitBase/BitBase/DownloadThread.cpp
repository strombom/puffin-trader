#include "pch.h"

#include "Logger.h"
#include "DateTime.h"
#include "DownloadThread.h"


DownloadThread::DownloadThread(manager_callback_done_t manager_callback_done) :
    state(DownloadState::idle), manager_callback_done(manager_callback_done)
{
    worker = std::make_unique<std::thread>(std::bind(&DownloadThread::worker_thread, this));
}

DownloadThread::~DownloadThread(void)
{
    shutdown();
    join();
}

void DownloadThread::shutdown(void)
{
    {
        auto slock = std::scoped_lock{ state_mutex };
        state = DownloadState::shutting_down;
    }
    download_start_condition.notify_all();
}

void DownloadThread::join(void) const
{
    logger.info("DownloadThread.join start");
    if (worker->joinable()) {
        if (state == DownloadState::idle) {
            logger.info("DownloadThread.join (idle)");
        } 
        else if (state == DownloadState::downloading) {
            logger.info("DownloadThread.join (downloading)");
        }
        else if (state == DownloadState::aborting) {
            logger.info("DownloadThread.join (aborting)");
        }
        else if (state == DownloadState::shutting_down) {
            logger.info("DownloadThread.join (shutting_down)");
        }

        try {
            worker->join();
            logger.info("DownloadThread.join done");
        }
        catch (...) {
            logger.info("DownloadThread.join fail");
        }
    }
    logger.info("DownloadThread.join end");
}

void DownloadThread::abort_download(void)
{
    auto slock = std::scoped_lock{ state_mutex };
    state = DownloadState::aborting;
}

bool DownloadThread::is_idle(void) const
{
    return state == DownloadState::idle;
}

bool DownloadThread::test_client_id(const std::string& client_id)
{
    auto slock = std::scoped_lock{ working_task_mutex };
    return state == DownloadState::downloading && working_task && working_task->get_client_id() == client_id;
}

void DownloadThread::assign_task(uptrDownloadTask new_task)
{
    auto slock = std::scoped_lock{ state_mutex };
    if (state == DownloadState::idle) {
        pending_task = std::move(new_task);
        state = DownloadState::pending;
        download_start_condition.notify_one();
    }
}

void DownloadThread::worker_thread(void)
{
    while (state != DownloadState::shutting_down) {
        {
            auto download_start_lock = std::unique_lock{ download_start_mutex };
            download_start_condition.wait(download_start_lock);
        }

        {
            auto slock = std::scoped_lock{ state_mutex, working_task_mutex };
            if (state == DownloadState::shutting_down) {
                break;
            }
            else if (state == DownloadState::pending) {
                state = DownloadState::downloading;
                working_task = std::move(pending_task);
            }
            else {
                continue;
            }
        }

        while (state == DownloadState::downloading) {
           auto curl = curl_easy_init();
            if (curl) {
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
                curl_easy_setopt(curl, CURLOPT_NOSIGNAL, TRUE);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
                curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, this);
                curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, download_progress_callback);
                curl_easy_setopt(curl, CURLOPT_URL, working_task->get_url().c_str());
                auto timer = Timer{};
                logger.info("DownloadThread::worker_thread download (%d) start", working_task->get_download_id());
                CURLcode res = curl_easy_perform(curl);
                {
                    auto state_lock = std::scoped_lock{ state_mutex };
                    if (res == CURLE_OK && state != DownloadState::aborting && state != DownloadState::shutting_down) {
                        auto task = uptrDownloadTask{};
                        {
                            auto task_lock = std::scoped_lock{ working_task_mutex };
                            task = std::move(working_task);
                        }
                        logger.info("DownloadThread::worker_thread download (%d) success (%d ms)", task->get_download_id(), timer.elapsed().count() / 1000);
                        manager_callback_done(std::move(task));
                        state = DownloadState::idle;
                    } else {
                        logger.info("DownloadThread::worker_thread download failed (%d ms)", timer.elapsed().count() / 1000);
                        working_task->clear_data();
                    }
                }
                curl_easy_cleanup(curl);
            }
        }
    }
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    ((DownloadThread*)arg)->working_task->append_data((const char*)ptr, (std::streamsize) count);
    return count;
}

size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow)
{
    if (((DownloadThread*)arg)->state == DownloadState::aborting || ((DownloadThread*)arg)->state == DownloadState::shutting_down) {
        logger.info("DownloadThread download_progress_callback aborting thread");
        return CURLE_ABORTED_BY_CALLBACK;
    }
    else {
        return CURLE_OK;
    }
}
