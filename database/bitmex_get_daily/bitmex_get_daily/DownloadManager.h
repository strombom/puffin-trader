#pragma once

#include <queue>

#include "DownloadManagerThread.h"

class DownloadManager {
public:
    DownloadManager(std::string _url_front, std::string _url_back);
    ~DownloadManager(void);

    void download(std::string _url_middle);

    void join(void);
    void download_done_callback(int thread_idx);
    void download_progress_callback(void);

private:
    static const int thread_max_count = 2;
    DownloadManagerThread threads[thread_max_count];
    int active_thread_count = 0;
    std::queue<std::string> download_url_queue;
    std::string url_front, url_back;
    
    int current_download_id = 0;

    bool start_download(std::string url);
};
