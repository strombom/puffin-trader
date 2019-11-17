
#include "Logger.h"
#include "DownloadThread.h"

#include "curl/curl.h"

#pragma warning(disable: 26812) // Disable enum warning for CURLcode


std::mutex download_tasks_mutex;
std::vector<std::shared_ptr<std::future<void>>> download_tasks;


DownloadThread::DownloadThread(void) :
    state(DownloadState::idle)
//const std::string& url, std::string client_id, std::string callback_arg, client_callback_done_t client_callback_done, manager_callback_done_t manager_callback_done) :
//    url(url), client_id(client_id), callback_arg(callback_arg), client_callback_done(client_callback_done), manager_callback_done(manager_callback_done),
//    download_count_progress(0), state(DownloadState::ready_to_start), download_task(NULL) //, download_thread(NULL)
{
    worker = std::make_unique<std::thread>(std::bind(&DownloadThread::worker_thread, this));
    

    /*
    download_data = std::make_shared<payload_t>();

    // Housekeeping
    for (auto&& download_task = download_tasks.begin(); download_task != download_tasks.end();) {
        if (!(*download_task)->valid()) {
            download_task = download_tasks.erase(download_task);
        }
        else {
            ++download_task;
        }
    }
    */
}

DownloadThread::~DownloadThread(void)
{
    //logger.info("DownloadThread::~DownloadThread (%s) (%s)", client_id.c_str(), callback_arg.c_str());

    //shutdown();
    //join();
}

bool DownloadThread::is_idle(void)
{
    return state == DownloadState::idle;
}

void DownloadThread::assign_task(uptrDownloadTask new_task)
{
    task = std::move(new_task);
    download_start_condition.notify_one();
}

void DownloadThread::worker_thread(void)
{
    while (true) {
        std::unique_lock<std::mutex> lock(state_mutex);
        download_start_condition.wait(lock);
        logger.info("Running once");



    }
}

/*
void DownloadThread::start(void)
{
    std::scoped_lock lock(state_mutex);

    state = DownloadState::downloading;

    auto task = std::make_shared<std::future<void>>(std::async(std::launch::async, &DownloadThread::download, this));
    download_tasks.push_back(std::move(task));

    //download_thread = new std::thread(std::bind(&DownloadThread::download, this));
    //download_task = new std::async<void>(std::launch::async, &DownloadThread::download, this);
}

void DownloadThread::shutdown(void)
{
    std::scoped_lock lock(state_mutex);

    //if (download_thread != NULL) {
    if (download_task->valid()) {
        state = DownloadState::aborting;
    }
}

void DownloadThread::join(void) const
{
    //if (download_thread != NULL) {
    //    download_thread->join();
    //}
    if (download_task->valid()) {
        download_task->wait();
    }
}

void DownloadThread::append_data(const std::byte* data, std::streamsize size)
{
    download_data->insert(download_data->end(), data, data + size);

    download_count_progress += (int)size;
    if (download_count_progress >= download_progress_size) {
        download_count_progress -= download_progress_size;
    }
}

bool DownloadThread::test_id(std::string _client_id, std::string _callback_arg) const
{
    return client_id == _client_id && callback_arg == _callback_arg;
}

bool DownloadThread::test_id(std::string _client_id) const
{
    return client_id == _client_id;
}

bool DownloadThread::is_ready_to_start(void)
{
    std::scoped_lock lock(state_mutex);
    return state == DownloadState::ready_to_start;
}

bool DownloadThread::is_aborting(void)
{
    std::scoped_lock lock(state_mutex);
    return state == DownloadState::aborting;
}

    */

/*
bool DownloadThread::is_finished(void)
{
    std::scoped_lock lock(state_mutex);
    return state == DownloadState::finished;
}
*/

/*
bool DownloadThread::has_data(void)
{
    std::scoped_lock lock(state_mutex);

    auto status = download_task->wait_for(std::chrono::seconds(0));
    return state == DownloadState::has_data; //&& status == std::future_status::ready;
}

void DownloadThread::pass_data_to_client(void)
{
    std::scoped_lock lock(state_mutex);

    logger.info("DownloadThread::pass_data_to_client (%s) (%s)", client_id.c_str(), callback_arg.c_str());

    client_callback_done(callback_arg, download_data);

    //download_thread->detach();
    //download_thread = NULL;
    //state = DownloadState::finished;
}

void DownloadThread::download(void)
{
    while (true) {
        CURL* curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
            curl_easy_setopt(curl, CURLOPT_NOSIGNAL, TRUE);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
            curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, this);
            curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, download_progress_callback);
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            CURLcode res = curl_easy_perform(curl);
            {
                std::scoped_lock lock(state_mutex);
                if (res == CURLE_OK && state != DownloadState::aborting) {
                    state = DownloadState::has_data;
                } else {
                    download_data->clear();
                }
            }
            curl_easy_cleanup(curl);
        }

        {
            std::scoped_lock lock(state_mutex);
            if (state == DownloadState::has_data) {
                manager_callback_done(client_id, callback_arg);
                return;

            } else if (state == DownloadState::aborting) {
                return;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        {
            std::scoped_lock lock(state_mutex); 
            if (state == DownloadState::aborting) {
                return;
            }
        }
    }
}

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
    ((DownloadThread*)arg)->append_data((const std::byte*)ptr, (std::streamsize) count);
    return count;
}

size_t download_progress_callback(void* arg, double dltotal, double dlnow, double ultotal, double ulnow)
{
    if (((DownloadThread*)arg)->is_aborting()) {
        return CURLE_ABORTED_BY_CALLBACK;
    } else {
        return CURLE_OK;
    }
}

*/

