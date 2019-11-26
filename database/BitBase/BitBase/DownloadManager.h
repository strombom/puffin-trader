#pragma once

#include "DownloadThread.h"

#include <queue>


class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    static std::shared_ptr<DownloadManager> create(void);

    void download(std::string url, std::string client_id, client_callback_done_t client_callback_done);
    void download_done_callback(uptrDownloadTask task);

    void abort_client(std::string client_id);
    void shutdown(void);
    void join(void);

private:
    static const int threads_count = 5;
    std::mutex threads_mutex;
    std::mutex download_done_mutex;
    std::mutex client_args_mutex;

    std::deque<uptrDownloadTask> pending_tasks;
    std::unordered_map<std::string, std::deque<uptrDownloadTask>> finished_tasks;
    std::unordered_map<std::string, int> next_download_id;
    std::unordered_map<std::string, int> expected_download_id;

    std::vector<sptrDownloadThread> threads;
    
    bool start_pending_downloads(void);
};

using sptrDownloadManager  = std::shared_ptr<DownloadManager>;
