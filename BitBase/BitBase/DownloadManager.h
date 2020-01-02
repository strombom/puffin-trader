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
    static constexpr auto threads_count = 5;
    std::vector<sptrDownloadThread> threads;
    std::mutex threads_mutex;

    std::mutex pending_tasks_mutex;
    std::mutex finished_tasks_mutex;
    std::condition_variable pending_tasks_condition;
    std::condition_variable finished_tasks_condition;

    std::deque<uptrDownloadTask> pending_tasks;
    std::deque<uptrDownloadTask> finished_tasks;
    std::unordered_map<std::string, int> next_download_id;
    std::unordered_map<std::string, int> expected_download_id;

    bool running;
    void pending_tasks_thread(void);
    void finished_tasks_thread(void);

    std::unique_ptr<std::thread> pending_tasks_thread_handle;
    std::unique_ptr<std::thread> finished_tasks_thread_handle;
};

using sptrDownloadManager = std::shared_ptr<DownloadManager>;
