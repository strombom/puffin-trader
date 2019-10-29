#include "DownloadThread.h"

#include "curl/curl.h"

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)

void DownloadThread::attach_signals(boost::function<void(int, int)> _signal_download_done,
                                           boost::function<void(void)>     _signal_download_progress,
                                           int _thread_idx)
{
    thread_idx = _thread_idx;
    signal_download_done.connect(_signal_download_done);
    signal_download_progress.connect(_signal_download_progress);
}

std::stringstream* DownloadThread::get_data(void)
{
    return &download_data;
}

void DownloadThread::start_download(const std::string& _url, int _download_id)
{
    download_id = _download_id;
    url = _url;
    restart_download();
}

void DownloadThread::restart_download(void)
{
    running = true;
    thread = new boost::thread(&DownloadThread::download_file, this);
    download_data.clear();
    download_count = 0;
    download_count_progress = 0;
}

bool DownloadThread::is_running(void)
{
    return running;
}

void DownloadThread::join(void)
{
    if (thread != NULL) {
        thread->join();
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

int DownloadThread::get_download_id(void)
{
    return download_id;
}

void DownloadThread::download_file(void)
{
    CURL* curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            download_data.clear();
        }
        curl_easy_cleanup(curl);
    }    
    running = false;
    signal_download_done(thread_idx, download_id);
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    DownloadThread* download_manager_thread = (DownloadThread*)arg;
    download_manager_thread->append_data((const char*)ptr, (std::streamsize) count);
    return count;
}
