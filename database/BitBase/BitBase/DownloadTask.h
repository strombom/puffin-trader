#pragma once

#include <functional>

using download_data_t = std::vector<char>;
using sptr_download_data_t = std::shared_ptr<download_data_t>;
using client_callback_done_t = std::function<void(std::string, sptr_download_data_t)>;


class DownloadTask {
public:
    DownloadTask(const std::string& url, std::string client_id, std::string client_arg, client_callback_done_t client_callback_done);
    ~DownloadTask(void);

    static std::unique_ptr<DownloadTask> create(const std::string& url, std::string client_id, std::string client_arg, client_callback_done_t client_callback_done);

    const std::string& get_url(void) const;
    const std::string& get_client_id(void) const;
    const std::string& get_client_arg(void) const;
    void clear_data(void);
    void append_data(const char* data, std::streamsize size);
    void run_client_callback(void);


private:
    const std::string url;
    const std::string client_id;
    const std::string client_arg;
    const client_callback_done_t client_callback_done;

    sptr_download_data_t download_data;

};

using uptrDownloadTask = std::unique_ptr<DownloadTask>;
