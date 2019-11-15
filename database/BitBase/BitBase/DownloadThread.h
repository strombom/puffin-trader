#pragma once

#include <future>

using namespace std::placeholders;  // for _1, _2, _3...
using payload_t = std::vector<std::byte>;
using sptr_payload_t = std::shared_ptr<payload_t>;
using client_callback_done_t = std::function<void(std::string, sptr_payload_t)>;
using manager_callback_done_t = std::function<void(std::string, std::string)>;


enum class DownloadState {
    waiting_for_start,
    downloading,
    aborting,
    success,
    finished
};

using DownloadStateAtomic = std::atomic<DownloadState>;

class DownloadThread {
public:
    DownloadThread(const std::string& url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done, manager_callback_done_t manager_callback_done);
    ~DownloadThread(void);

    void start(void);
    void shutdown(void);

    DownloadState get_state(void) const;
    bool test_id(std::string _client_id, std::string _callback_arg) const;
    bool test_id(std::string _client_id) const;
    void join(void) const;

    friend size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);

private:
    static const int download_progress_size = (int)10e5;

    DownloadStateAtomic state;
    const client_callback_done_t client_callback_done;
    const manager_callback_done_t manager_callback_done;
    int download_count_progress;
    std::mutex state_mutex;

    const std::string url;
    const std::string client_id;
    const std::string callback_arg;
    std::future<void> download_task;
    sptr_payload_t download_data;
    
    void download(void);
    void append_data(const std::byte* data, std::streamsize size);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);

using uptrDownloadThread = std::unique_ptr<DownloadThread>;
