#include "DownloadManager.h"

#include "curl/curl.h"

#pragma warning (disable : 26444)

DownloadManager::DownloadManager(std::string _url_front, std::string _url_back)
{
    url_front = _url_front;
    url_back = _url_back;

    curl_global_init(CURL_GLOBAL_ALL);

    for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {

        threads[thread_idx].attach_signals(boost::bind(&DownloadManager::download_done_callback, this, _1),
                                           boost::bind(&DownloadManager::download_progress_callback, this),
                                           thread_idx);
    }
}

DownloadManager::~DownloadManager(void)
{
    curl_global_cleanup();
}

void DownloadManager::download(std::string _url_middle)
{
    std::string url = url_front + _url_middle + url_back;
    if (!start_download(url)) {
        download_url_queue.push(url);
    }
}

bool DownloadManager::start_download(std::string url)
{
    int thread_idx;
    for (thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
        if (!threads[thread_idx].is_running()) {
            break;
        }
    }

    if (thread_idx == thread_max_count) {
        // No free thread
        return false;
    }

    threads[thread_idx].start_download(url);
    active_thread_count++;
    return true;
}

void DownloadManager::join(void)
{
    while (active_thread_count > 0) {
        boost::posix_time::seconds seconds(1);
        boost::this_thread::sleep(seconds);
    }
}

void DownloadManager::download_done_callback(int thread_idx)
{
    std::stringstream* data = threads[thread_idx].get_data();
    data->seekg(0, std::stringstream::end);
    unsigned int length = (int) data->tellg();

    printf("\nDownload done %d, length %d\n", thread_idx, length);

    if (length == 0) {
        threads[thread_idx].restart_download();
        return;
    }

    active_thread_count--;
    if (download_url_queue.size() > 0) {
        if (start_download(download_url_queue.front())) {
            download_url_queue.pop();
        }
    }
}

void DownloadManager::download_progress_callback(void)
{
    bool first = true;
    printf("\33[2K\r");
    for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
        if (threads[thread_idx].is_running()) {
            if (!first) {
                printf("  ");
            }
            else {
                first = false;
            }

            printf("Progress(%d) % 3.1f MB", thread_idx, threads[thread_idx].get_progress());
        }
    }
    fflush(stdout);
}
