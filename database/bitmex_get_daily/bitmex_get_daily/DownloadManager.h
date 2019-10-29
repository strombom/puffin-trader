#pragma once

#include <queue>

#include "DownloadThread.h"

class DownloadManager {
public:
    DownloadManager(std::string _url_front, std::string _url_back);
    ~DownloadManager(void);

    void download(std::string _url_middle);

    void join(void);
    void download_done_callback(int thread_idx, int download_id);
    void download_progress_callback(void);

private:
    static const int thread_max_count = 2;
    DownloadThread threads[thread_max_count];
   // std::queue<DownloadThread> threads;
    int active_thread_count = 0;
    std::queue<std::string> download_url_queue;
    std::string url_front, url_back;
    
    int next_download_id = 0;
    int last_finished_download_id = -1;

    bool start_download(std::string url);
};
