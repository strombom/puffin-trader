#pragma once

#include "boost/signals2.hpp"
#include "boost/thread.hpp"

class DownloadManagerThread {
public:

    void attach_signals(boost::function<void(int)>  _signal_download_done,
        boost::function<void(void)> _signal_download_progress,
        int _thread_idx);
    void start_download(const std::string& url);
    void restart_download(void);
    bool is_running(void);
    void join(void);
    void append_data(const char* data, std::streamsize size);
    float get_progress(void);
    std::stringstream* get_data(void);

private:
    int thread_idx = -1;
    bool running = false;
    int download_count = 0;
    int download_count_progress = 0;
    static const int download_progress_size = (int)10e5;
    std::stringstream download_data;
    std::string url;

    boost::thread* thread;

    boost::signals2::signal<void(int)>  signal_download_done;
    boost::signals2::signal<void(void)> signal_download_progress;

    void download_file(void);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
