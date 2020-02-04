#include "pch.h"

#include "DownloadTask.h"

#include <string>

DownloadTask::DownloadTask(const std::string& url, std::string client_id, int download_id, client_callback_done_t client_callback_done) :
    url(url), client_id(client_id), download_id(download_id), client_callback_done(client_callback_done)
{
    download_data = std::make_shared<download_data_t>();
}

DownloadTask::~DownloadTask(void)
{

}

std::unique_ptr<DownloadTask> DownloadTask::create(const std::string& url, std::string client_id, int download_id, client_callback_done_t client_callback_done)
{
    return std::make_unique<DownloadTask>(url, client_id, download_id, client_callback_done);
}


const std::string& DownloadTask::get_url(void) const
{
    return url;
}

const std::string& DownloadTask::get_client_id(void) const
{
    return client_id;
}

const int DownloadTask::get_download_id(void) const
{
    return download_id;
}

void DownloadTask::clear_data(void)
{
    download_data->clear();
}

void DownloadTask::append_data(const char* data, std::streamsize size)
{
    download_data->insert(download_data->end(), data, data + size);
}

void DownloadTask::call_client_callback(void)
{
    client_callback_done(download_data);
}
