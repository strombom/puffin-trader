#include "DownloadTask.h"

#include <string>

DownloadTask::DownloadTask(const std::string& url, std::string client_id, std::string client_arg, client_callback_done_t client_callback_done) :
    url(url), client_id(client_id), client_arg(client_arg), client_callback_done(client_callback_done)
{
    download_data = std::make_shared<download_data_t>();
}

DownloadTask::~DownloadTask(void)
{

}

std::unique_ptr<DownloadTask> DownloadTask::create(const std::string& url, std::string client_id, std::string client_arg, client_callback_done_t client_callback_done)
{
    return std::make_unique<DownloadTask>(url, client_id, client_arg, client_callback_done);
}


const std::string& DownloadTask::get_url(void) const
{
    return url;
}

const std::string& DownloadTask::get_client_id(void) const
{
    return client_id;
}

const std::string& DownloadTask::get_client_arg(void) const
{
    return client_arg;
}

void DownloadTask::clear_data(void)
{
    download_data->clear();
}

void DownloadTask::append_data(const std::byte* data, std::streamsize size)
{
    download_data->insert(download_data->end(), data, data + size);
}

void DownloadTask::run_client_callback(void)
{
    client_callback_done(client_arg, download_data);
}
