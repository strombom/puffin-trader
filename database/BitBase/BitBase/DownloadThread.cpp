
#include "Logger.h"
#include "DownloadThread.h"

#include "curl/curl.h"


DownloadThread::DownloadThread(const std::string& url, std::string client_id, std::string callback_arg, manager_callback_done_t manager_callback_done) :
    url(url), client_id(client_id), callback_arg(callback_arg), manager_callback_done(manager_callback_done),
    download_count_progress(0), state(DownloadState::waiting_for_start)
{
    download_data = std::make_shared<payload_t>();
}

void DownloadThread::start(void)
{
    logger.info("Thread start (%s) (%s)", client_id.c_str(), callback_arg.c_str());

    state = DownloadState::downloading;
    download_task = std::async(&DownloadThread::download, this);
}

void DownloadThread::shutdown(void)
{
    std::scoped_lock lock(state_mutex);

    if (download_task.valid()) {
        state = DownloadState::aborting;
    }
}

void DownloadThread::join(void) const
{
    if (download_task.valid()) {
        download_task.wait();
    }
}

void DownloadThread::append_data(const std::byte* data, std::streamsize size)
{
    download_data->insert(download_data->end(), data, data + size);

    download_count_progress += (int)size;
    if (download_count_progress >= download_progress_size) {
        download_count_progress -= download_progress_size;
    }
}

DownloadState DownloadThread::get_state(void) const
{
    return state;
}

bool DownloadThread::test_id(std::string _client_id, std::string _callback_arg) const
{
    return client_id == _client_id && callback_arg == _callback_arg;
}

bool DownloadThread::test_id(std::string _client_id) const
{
    return client_id == _client_id;
}

void DownloadThread::download(void)
{
    while (true) {
        CURL* curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
            curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, this);
            curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, download_progress_callback);
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            CURLcode res = curl_easy_perform(curl);
            {
                std::scoped_lock lock(state_mutex);
                if (res == CURLE_OK && state != DownloadState::aborting) {
                    state = DownloadState::success;
                } else {
                    download_data->clear();
                }
            }
            curl_easy_cleanup(curl);
        }

        {
            std::scoped_lock lock(state_mutex);
            if (state == DownloadState::success) {
                manager_callback_done(client_id, callback_arg, download_data);
                return;

            } else if (state == DownloadState::aborting) {
                return;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        {
            std::scoped_lock lock(state_mutex); 
            if (state == DownloadState::aborting) {
                return;
            }
        }
    }
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    ((DownloadThread*)arg)->append_data((const std::byte*)ptr, (std::streamsize) count);
    return count;
}

size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow)
{
    if (((DownloadThread*)arg)->get_state() == DownloadState::aborting) {
        return CURLE_ABORTED_BY_CALLBACK;
    } else {
        return CURLE_OK;
    }
}
