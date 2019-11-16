#include "DownloadTask.h"

#include <string>

DownloadTask::DownloadTask(const std::string& url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done) :
    url(url), client_id(client_id), callback_arg(callback_arg), client_callback_done(client_callback_done)
{

}

DownloadTask::~DownloadTask(void)
{

}
