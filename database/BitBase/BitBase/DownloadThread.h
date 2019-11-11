#pragma once

#include "boost/signals2.hpp"
#include "boost/thread.hpp"


using manager_callback_done_t = boost::function<void(std::string, std::string, std::shared_ptr<std::vector<std::byte>>)>;

enum class DownloadState {
    idle,
    downloading,
    aborting,
    failed,
    success
};

class DownloadThread {
public:
    DownloadThread(const std::string& _url, std::string client_id, std::string callback_arg, manager_callback_done_t _manager_callback_done);
    //DownloadThread(const std::string& _url,
    //    boost::function<void(void)> _signal_download_done,
    //    boost::function<void(void)> _signal_download_progress);

    //void DownloadManager::download(std::string url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done)
    //{
    //    std::unique_ptr<DownloadThread> download_thread(new DownloadThread(url, client_id, callback_arg, client_callback_done,
    //        boost::bind(&DownloadManager::download_done_callback, this)));
    //    //                                                                       boost::bind(&DownloadManager::download_done_callback, this),
    //    //                                                                       boost::bind(&DownloadManager::download_progress_callback, this)));

    //float get_progress(void);
    DownloadState get_state(void);

    void start(void);
    void shutdown(void);
    void join(void);

private:
    int download_count_progress = 0;
    static const int download_progress_size = (int)10e5;

    std::string url;
    std::string client_id;
    std::shared_ptr<std::vector<std::byte>> download_data;

    DownloadState state = DownloadState::idle;

    boost::thread* download_thread;
    manager_callback_done_t manager_callback_done;

    boost::signals2::signal<void(void)> signal_download_done;
    boost::signals2::signal<void(void)> signal_download_progress;

    void download_file(void);
    void restart_download(void);
    void append_data(const std::byte* data, std::streamsize size);
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);
size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow);
