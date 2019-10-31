#pragma once

#include <queue>

#include "DownloadThread.h"

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    void download(std::string url);

    void shutdown(void);
    void join(void);
    void download_done_callback(void);
    void download_progress_callback(void);

private:
    std::queue<std::string> download_url_queue;
    static const int thread_max_count = 4;
    std::deque<std::unique_ptr<DownloadThread>> threads;
    
    bool start_download(std::string url);
};
