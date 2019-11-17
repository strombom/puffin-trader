
#include "Logger.h"
#include "DownloadThread.h"

#include "curl/curl.h"

#pragma warning(disable: 26812) // Disable enum warning for CURLcode


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
    std::scoped_lock lock(state_mutex);
    state = DownloadState::shutting_down;
}

void DownloadThread::join(void) const
{
    if (worker->joinable()) {
        worker->join();
    }
}

void DownloadThread::abort(void)
{
    std::scoped_lock lock(state_mutex);
    state = DownloadState::aborting;
}

bool DownloadThread::is_idle(void) const
{
    return state == DownloadState::idle;
}

bool DownloadThread::test_client_id(std::string client_id) const
{
    return state == DownloadState::downloading && task && task->get_client_id() == client_id;
}

void DownloadThread::assign_task(uptrDownloadTask new_task)
{
    std::scoped_lock slock(state_mutex);
    state = DownloadState::downloading;
    task = std::move(new_task);
    download_start_condition.notify_one();
}

void DownloadThread::worker_thread(void)
{
    while (state != DownloadState::shutting_down) {

        std::unique_lock<std::mutex> download_start_lock(download_start_mutex);
        download_start_condition.wait(download_start_lock);

        while (state == DownloadState::downloading) {
            CURL* curl = curl_easy_init();
            if (curl) {
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
                curl_easy_setopt(curl, CURLOPT_NOSIGNAL, TRUE);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
                curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
                curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, this);
                curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, download_progress_callback);
                curl_easy_setopt(curl, CURLOPT_URL, task->get_url().c_str());
                CURLcode res = curl_easy_perform(curl);
                {
                    std::scoped_lock slock(state_mutex);
                    if (res == CURLE_OK && state != DownloadState::aborting && state != DownloadState::shutting_down) {
                        manager_callback_done(std::move(task));
                        state = DownloadState::idle;
                    } else {
                        task->clear_data();
                    }
                }
                curl_easy_cleanup(curl);
            }
        }
    }
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    ((DownloadThread*)arg)->task->append_data((const std::byte*)ptr, (std::streamsize) count);
    return count;
}

size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow)
{
    if (((DownloadThread*)arg)->state == DownloadState::aborting || ((DownloadThread*)arg)->state == DownloadState::shutting_down) {
        return CURLE_ABORTED_BY_CALLBACK;
    }
    else {
        return CURLE_OK;
    }
}
