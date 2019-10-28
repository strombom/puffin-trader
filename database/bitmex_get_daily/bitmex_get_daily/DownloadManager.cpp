#include "DownloadManager.h"

#include "curl/curl.h"

#pragma warning (disable : 26444)

DownloadManager::DownloadManager(void)
{
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

void DownloadManager::download(const boost::gregorian::date& date)
{
    if (!start_download(date)) {
        download_queue.push(date);
    }
}

bool DownloadManager::start_download(const boost::gregorian::date& date)
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

    std::string url = make_url(date);
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
    printf("\nDownload done %d\n", thread_idx);
    active_thread_count--;
    if (download_queue.size() > 0) {
        if (start_download(download_queue.front())) {
            download_queue.pop();
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

std::string DownloadManager::make_url(boost::gregorian::date date)
{
    std::stringstream url;
    url.imbue(std::locale(std::cout.getloc(), new boost::date_time::date_facet < boost::gregorian::date, char>("%Y%m%d")));
    url << "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/" << date << ".csv.gz";
    return url.str();
}
