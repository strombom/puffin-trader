#pragma once

#include <stdio.h>
#include <tuple>
#include <queue>

#include "DownloadThread.h"


using client_callback_done_t = std::function<void(std::string, std::shared_ptr<std::vector<std::byte>>)>;
using uptrDownloadThread = std::unique_ptr<DownloadThread>;


class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    void download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done);

    void shutdown(void);
    void join(void);
    
    void download_done_callback(std::string client_id, std::string callback_arg, std::shared_ptr<std::vector<std::byte>> payload);

private:
    int active_threads_count = 0;
    static const int max_active_threads_count = 4;

    std::deque<uptrDownloadThread> threads;

    void start_next(void);
};
