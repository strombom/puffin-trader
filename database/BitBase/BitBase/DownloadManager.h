#pragma once

#include <stdio.h>
#include <queue>

#include "DownloadThread.h"

class DownloadHandle {
public:
    //DownloadHandle(void);

};

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    void download(std::string url, std::string callback_arg, client_callback_done_t client_callback_done);

    void shutdown(void);
    void join(void);
    
    //void download_done_callback(void);
    //void download_progress_callback(void);

private:
    //std::queue<std::string> threads;
    static const int max_active_thread_count = 4;
    std::deque<std::unique_ptr<DownloadThread>> threads;

    void tick(void);
};
