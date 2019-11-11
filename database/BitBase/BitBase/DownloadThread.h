#pragma once

#include "boost/signals2.hpp"
#include "boost/thread.hpp"


using client_callback_done_t = boost::function<void(std::shared_ptr<std::vector<std::byte>>, std::string)>;

enum class DownloadState {
    idle,
    downloading,
    aborting,
    failed,
    success
};

class DownloadThread {
public:
    DownloadThread(const std::string& _url, std::string callback_arg, client_callback_done_t _client_callback_done);
    //DownloadThread(const std::string& _url,
    //    boost::function<void(void)> _signal_download_done,
    //    boost::function<void(void)> _signal_download_progress);

    float get_progress(void);
    DownloadState get_state(void);
    //std::stringstream* get_data(void);
    void join(void);
    void shutdown(void);
    std::string get_url(void);

private:
    int download_count_progress = 0;
    static const int download_progress_size = (int)10e5;

    std::string url;
    std::shared_ptr<std::vector<std::byte>> download_data;

    DownloadState state = DownloadState::idle;

    boost::thread* download_thread;

    boost::signals2::signal<void(void)> signal_download_done;
    boost::signals2::signal<void(void)> signal_download_progress;

    void download_file(void);
    void restart_download(void);
    void append_data(const std::byte* data, std::streamsize size);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);
