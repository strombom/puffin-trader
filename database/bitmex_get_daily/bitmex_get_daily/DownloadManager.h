#pragma once

#include <queue>
#include "boost/date_time/gregorian/gregorian.hpp"

#include "DownloadManagerThread.h"

class DownloadManager {
public:
    DownloadManager(void);
    ~DownloadManager(void);

    void download(const boost::gregorian::date& date);

    void join(void);
    void download_done_callback(int thread_idx);
    void download_progress_callback(void);

private:
    static const int thread_max_count = 2;
    DownloadManagerThread threads[thread_max_count];
    int active_thread_count = 0;
    std::queue<boost::gregorian::date> download_queue;

    bool start_download(const boost::gregorian::date& date);
    std::string make_url(boost::gregorian::date date);
};
