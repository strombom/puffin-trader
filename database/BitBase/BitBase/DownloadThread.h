#pragma once

#include "DownloadTask.h"

#include <future>

using namespace std::placeholders;  // for _1, _2, _3...
using manager_callback_done_t = std::function<void(uptrDownloadTask)>;


enum class DownloadState {
    idle,
    downloading,
    aborting,
    shutting_down
};

using DownloadStateAtomic = std::atomic<DownloadState>;

class DownloadThread {
public:
    DownloadThread(manager_callback_done_t manager_callback_done);
    ~DownloadThread(void);

    void abort(void);       // Abort download
    void shutdown(void);    // Abort download and exit thread
    void join(void) const;  // Join thread

    bool is_idle(void) const;
    bool test_client_id(std::string client_id) const;

    void assign_task(uptrDownloadTask new_task);

    friend size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
    friend size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);

private:
    DownloadState state;

    uptrDownloadTask task;

    std::mutex state_mutex;
    std::mutex download_start_mutex;
    std::condition_variable download_start_condition;
    std::unique_ptr<std::thread> worker;

    void worker_thread(void);

    const manager_callback_done_t manager_callback_done;
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);

using sptrDownloadThread = std::shared_ptr<DownloadThread>;
