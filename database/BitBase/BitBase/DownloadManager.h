#pragma once

#include "DownloadThread.h"

#include <queue>


using client_callback_done_t = std::function<void(std::string, std::shared_ptr<std::vector<std::byte>>)>;

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    static std::shared_ptr<DownloadManager> create(void);

    void download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done);
    void abort(std::string client_id);

    void shutdown(void);
    void join(void);
    
    void download_done_callback(std::string client_id, std::string callback_arg, payload_t payload);

private:
    int active_threads_count = 0;
    static const int max_active_threads_count = 4;

    std::deque<uptrDownloadThread> threads;

    void start_next(void);
};

using sptrDownloadManager  = std::shared_ptr<DownloadManager>;
