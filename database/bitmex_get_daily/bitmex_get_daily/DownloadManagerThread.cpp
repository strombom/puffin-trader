#include "DownloadManagerThread.h"

#include "curl/curl.h"

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)

void DownloadManagerThread::attach_signals(boost::function<void(int)>  _signal_download_done,
    boost::function<void(void)> _signal_download_progress,
    int _thread_idx)
{
    thread_idx = _thread_idx;
    signal_download_done.connect(_signal_download_done);
    signal_download_progress.connect(_signal_download_progress);
}

void DownloadManagerThread::start_download(const std::string& url)
{
    running = true;
    thread = new boost::thread(&DownloadManagerThread::download_file, this, url);
    download_count = 0;
    download_count_progress = 0;
}

bool DownloadManagerThread::is_running(void)
{
    return running;
}

void DownloadManagerThread::join(void)
{
    if (thread != NULL) {
        thread->join();
    }
}

void DownloadManagerThread::append_data(const char* data, std::streamsize size)
{
    download_count += (int)size;
    download_count_progress += (int)size;
    if (download_count_progress >= (int)10e5) {
        download_count_progress -= (int)10e5;
        signal_download_progress();
    }
}

float DownloadManagerThread::get_progress(void)
{
    return (float)(download_count / 10e6);
}

void DownloadManagerThread::download_file(const std::string& url)
{
    CURL* curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        CURLcode res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            //printf(", failed.\n");
            //printf("CURL error: %s\n", curl_easy_strerror(res));
            download_data.clear();
        }
        else {
            //printf(", OK.\n");
        }

        curl_easy_cleanup(curl);
    }

    download_data.seekg(0, std::stringstream::end);
    unsigned int length = (int)download_data.tellg();

    running = false;
    signal_download_done(thread_idx);
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    DownloadManagerThread* download_manager_thread = (DownloadManagerThread*)arg;
    download_manager_thread->append_data((const char*)ptr, (std::streamsize) count);
    return count;
}
