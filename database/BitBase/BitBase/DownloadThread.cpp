#include "DownloadThread.h"

#include "curl/curl.h"

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)


DownloadThread::DownloadThread(const std::string& url, std::string client_id, std::string callback_arg, manager_callback_done_t manager_callback_done) :
    url(url), client_id(client_id), callback_arg(callback_arg), manager_callback_done(manager_callback_done)
{
    state = DownloadState::waiting_for_start;
    download_data->clear();
    download_count_progress = 0;
}

void DownloadThread::start(void)
{
    state = DownloadState::downloading;
    download_thread_handle = new std::thread(&DownloadThread::download_thread, this);
}

void DownloadThread::shutdown(void)
{
    if (download_thread_handle != NULL) {
        state = DownloadState::aborting;
        download_thread_handle->join();
    }
}

void DownloadThread::join(void) const
{
    if (download_thread_handle != NULL) {
        download_thread_handle->join();
    }
}

void DownloadThread::append_data(const std::byte* data, std::streamsize size)
{
    download_data->insert(download_data->end(), (const std::byte*)data, (const std::byte*) (data + size));

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

void DownloadThread::download_thread(void)
{
    while (state != DownloadState::success) {
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
            if (res == CURLE_OK) {
                state = DownloadState::success;
            }
            else {
                download_data->clear();
            }
            curl_easy_cleanup(curl);
        }

        if (state == DownloadState::aborting) {
            break;

        } else if (state != DownloadState::success) {
            state = DownloadState::downloading;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }

    manager_callback_done(client_id, callback_arg, download_data);
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    DownloadThread* download_manager_thread = (DownloadThread*)arg;
    download_manager_thread->append_data((const std::byte*)ptr, (std::streamsize) count);
    return count;
}

size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow)
{
    DownloadThread* download_manager_thread = (DownloadThread*)arg;
    if (download_manager_thread->get_state() == DownloadState::aborting) {
        return CURLE_ABORTED_BY_CALLBACK;
    }
    return CURLE_OK;
}
