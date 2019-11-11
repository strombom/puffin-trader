#pragma once

#include <stdio.h>
#include <tuple>
#include <queue>

#include "DownloadThread.h"


using client_callback_done_t = boost::function<void(std::string, std::shared_ptr<std::vector<std::byte>>)>;

class DownloadHandle {
public:
    //DownloadHandle(void);

};

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    void download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done);

    void shutdown(void);
    void join(void);
    
    void download_done_callback(std::string client_id, std::string callback_arg, std::shared_ptr<std::vector<std::byte>> payload);
    //void download_progress_callback(void);

private:
    //std::queue<std::string> threads;
    static const int max_active_thread_count = 4;

    std::deque<std::tuple<std::string, std::unique_ptr<DownloadThread>>> threads;
    //std::map<std::string, std::deque<std::unique_ptr<DownloadThread>>> threads;

    void start_next(void);
};
