#pragma once

#include "DownloadThread.h"

#include <queue>



class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    static std::shared_ptr<DownloadManager> create(void);

    void download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done);
    void abort(std::string client_id);

    void shutdown(void);
    void join(void);

    void download_done_callback(std::string client_id, std::string callback_arg);

private:
    int active_threads_count = 0;
    static const int max_active_threads_count = 4;
    std::mutex threads_mutex;

    std::deque<uptrDownloadThread> threads;

    void start_next(void);
};

using sptrDownloadManager  = std::shared_ptr<DownloadManager>;
