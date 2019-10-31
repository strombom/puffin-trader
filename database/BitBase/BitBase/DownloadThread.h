#pragma once

#include "boost/signals2.hpp"
#include "boost/thread.hpp"


enum class DownloadState {
    idle,
    downloading,
    aborting,
    failed,
    success
};

class DownloadThread {
public:
    void attach_signals(boost::function<void(void)> _signal_download_done,
                        boost::function<void(void)> _signal_download_progress);
    void start_download(const std::string& url);
    void restart_download(void);
    void append_data(const char* data, std::streamsize size);
    float get_progress(void);
    DownloadState get_state(void);
    std::stringstream* get_data(void);
    void join(void);
    void shutdown(void);
    std::string get_url(void);

private:
    int download_count = 0;
    int download_count_progress = 0;
    static const int download_progress_size = (int)10e5;

    std::string url;
    std::stringstream download_data;

    DownloadState state = DownloadState::idle;

    boost::thread* download_thread;

    boost::signals2::signal<void(void)> signal_download_done;
    boost::signals2::signal<void(void)> signal_download_progress;

    void download_file(void);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);
