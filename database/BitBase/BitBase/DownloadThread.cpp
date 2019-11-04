#include "DownloadThread.h"

#include "curl/curl.h"

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)

void DownloadThread::attach_signals(boost::function<void(void)> _signal_download_done,
                                    boost::function<void(void)> _signal_download_progress)
{
    signal_download_done.connect(_signal_download_done);
    signal_download_progress.connect(_signal_download_progress);
}

void DownloadThread::start_download(const std::string& _url)
{
    url = _url;
    restart_download();
}

void DownloadThread::restart_download(void)
{
    state = DownloadState::downloading;
    download_thread = new boost::thread(&DownloadThread::download_file, this);
    download_data.clear();
    download_count = 0;
    download_count_progress = 0;
}

std::string DownloadThread::get_url(void)
{
    return url;
}

void DownloadThread::join(void)
{
    if (download_thread != NULL) {
        download_thread->join();
    }
}

void DownloadThread::shutdown(void)
{
    if (download_thread != NULL) {
        state = DownloadState::aborting;
        download_thread->join();
    }
}

void DownloadThread::append_data(const char* data, std::streamsize size)
{
    // Add data to download data stream
    download_data.write((const char*)data, (std::streamsize) size);

    download_count += (int)size;
    download_count_progress += (int)size;
    if (download_count_progress >= (int)10e5) {
        download_count_progress -= (int)10e5;
        signal_download_progress();
    }
}

float DownloadThread::get_progress(void)
{
    return (float)(download_count / 10e6);
}

DownloadState DownloadThread::get_state(void)
{
    return state;
}

std::stringstream* DownloadThread::get_data(void)
{
    return &download_data;
}

void DownloadThread::download_file(void)
{
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
        } else {
            download_data.clear();
        }
        curl_easy_cleanup(curl);
    }

    if (state != DownloadState::success) {
        state = DownloadState::failed;
    }
    signal_download_done();
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    DownloadThread* download_manager_thread = (DownloadThread*)arg;
    download_manager_thread->append_data((const char*)ptr, (std::streamsize) count);
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
