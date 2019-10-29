#pragma once

#include <queue>

#include "DownloadThread.h"

class DownloadManager {
public:
    DownloadManager(std::string _url_front, std::string _url_back);
    ~DownloadManager(void);

    void download(std::string _url_middle);

    void join(void);
    void download_done_callback(void);
    void download_progress_callback(void);

private:
    std::queue<std::string> download_url_queue;
    static const int thread_max_count = 4;
    std::deque<std::unique_ptr<DownloadThread>> threads;
    std::string url_front, url_back;
    
    bool start_download(std::string url);
};
