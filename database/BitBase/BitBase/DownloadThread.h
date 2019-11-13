#pragma once

#include <thread>
#include <functional>

using namespace std::placeholders;  // for _1, _2, _3...
using manager_callback_done_t = std::function<void(std::string, std::string, std::shared_ptr<std::vector<std::byte>>)>;


enum class DownloadState {
    idle,
    waiting_for_start,
    downloading,
    aborting,
    success
};

class DownloadThread {
public:
    DownloadThread(const std::string& url, std::string client_id, std::string callback_arg, manager_callback_done_t manager_callback_done);

    void start(void);
    void shutdown(void);

    DownloadState get_state(void) const;
    bool test_id(std::string _client_id, std::string _callback_arg) const;
    void join(void) const;

    friend size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);

private:
    DownloadState state = DownloadState::idle;
    std::thread* download_thread_handle = NULL;
    const manager_callback_done_t manager_callback_done;
    int download_count_progress = 0;
    static const int download_progress_size = (int)10e5;

    const std::string url;
    const std::string client_id;
    const std::string callback_arg;
    std::shared_ptr<std::vector<std::byte>> download_data;
    
    void download_thread(void);
    void append_data(const std::byte* data, std::streamsize size);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);
