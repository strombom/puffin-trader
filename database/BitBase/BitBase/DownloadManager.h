#pragma once

#include "DownloadThread.h"

#include <queue>

class DownloadTasks {

};

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    static std::shared_ptr<DownloadManager> create(void);

    void download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done);
    void abort(std::string client_id);

    void shutdown(void);
    void join(void);

    //void download_done_callback(std::string client_id, std::string callback_arg);

private:
    static const int threads_count = 3;
    //std::mutex threads_mutex;

    std::deque<uptrDownloadTask> pending_tasks;
    std::unordered_map<std::string, std::deque<uptrDownloadTask>> running_tasks;

    std::vector<uptrDownloadThread> threads;
    
    //void manage_threads(void);
};

using sptrDownloadManager  = std::shared_ptr<DownloadManager>;
