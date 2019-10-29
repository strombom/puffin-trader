#include "DownloadManager.h"

#include "curl/curl.h"

#pragma warning (disable : 26444)

DownloadManager::DownloadManager(std::string _url_front, std::string _url_back)
{
    url_front = _url_front;
    url_back = _url_back;

    curl_global_init(CURL_GLOBAL_ALL);
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
    if (threads.size() == thread_max_count) {
        return false;
    }

    std::unique_ptr<DownloadThread> download_thread(new DownloadThread);
    download_thread->attach_signals(boost::bind(&DownloadManager::download_done_callback, this),
                                    boost::bind(&DownloadManager::download_progress_callback, this));
    download_thread->start_download(url);

    threads.push_back(std::move(download_thread));
    return true;
}

void DownloadManager::join(void)
{
    while (threads.size() > 0) {
        boost::posix_time::seconds seconds(1);
        boost::this_thread::sleep(seconds);
    }
}

void DownloadManager::download_done_callback()
{
    while (threads.size() > 0 && threads[0]->get_state() == DownloadState::success)
    {
        printf("\nDone (%s)\n", threads[0]->get_url().c_str());
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
}

void DownloadManager::download_progress_callback(void)
{
    printf("\33[2K\r");
    for (int idx = 0; idx < threads.size(); idx++) {

        if (threads[idx]->get_state() == DownloadState::downloading) {
            printf("Progress % 3.1f MB ", threads[idx]->get_progress());

        } else if (threads[idx]->get_state() == DownloadState::success) {
            printf("Progress done ");

        } else if (threads[idx]->get_state() == DownloadState::failed) {
            printf("Progress failed ");
        }
    }
    fflush(stdout);
}
