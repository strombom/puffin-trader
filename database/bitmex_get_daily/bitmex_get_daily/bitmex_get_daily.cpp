// bitmex_get_daily.cpp
//

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)

#include <iostream>
#include <sstream>
#include <string>
#include <queue>

#include "curl/curl.h"
#include "boost/bind.hpp"
#include "boost/thread.hpp"
#include "boost/signals2.hpp"
#include "boost/lockfree/queue.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg);


class DownloadManagerThread {
public:

	void attach_signals(boost::function<void(int)>  _signal_download_done,
						boost::function<void(void)> _signal_download_progress,
						int _thread_idx)
	{
		thread_idx = _thread_idx;
		signal_download_done.connect(_signal_download_done);
		signal_download_progress.connect(_signal_download_progress);
	}

	void start_download(const std::string &url)
	{
		running = true;
		thread = new boost::thread(&DownloadManagerThread::download_file, this, url);
		download_count = 0;
		download_count_progress = 0;
	}

	bool is_running(void)
	{
		return running;
	}

	void join(void)
	{
		if (thread != NULL) {
			thread->join();
		}
	}

	void append_data(const char* data, std::streamsize size)
	{
		download_count += (int)size;
		download_count_progress += (int)size;
		if (download_count_progress >= (int)10e5) {
			download_count_progress -= (int)10e5;
			signal_download_progress();
		}
	}

	float get_progress(void)
	{
		return (float) (download_count / 10e6);
	}


private:
	int thread_idx = -1;
	bool running = false;
	int download_count = 0;
	int download_count_progress = 0;
	static const int download_progress_size = (int) 10e5;
	std::stringstream download_data;

	boost::thread* thread;

	boost::signals2::signal<void(int)>  signal_download_done;
	boost::signals2::signal<void(void)> signal_download_progress;

	void download_file(const std::string& url)
	{
		CURL* curl = curl_easy_init();
		if (curl) {
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
			curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

			CURLcode res = curl_easy_perform(curl);

			if (res != CURLE_OK) {
				//printf(", failed.\n");
				//printf("CURL error: %s\n", curl_easy_strerror(res));
				download_data.clear();
			}
			else {
				//printf(", OK.\n");
			}

			curl_easy_cleanup(curl);
		}

		download_data.seekg(0, std::stringstream::end);
		unsigned int length = (int)download_data.tellg();
		
		running = false;
		signal_download_done(thread_idx);
	}
};

size_t download_file_callback(void* ptr, size_t size, size_t count, void* arg)
{
	DownloadManagerThread* download_manager_thread = (DownloadManagerThread*)arg;
	download_manager_thread->append_data((const char*)ptr, (std::streamsize) count);
	return count;
}

class DownloadManager {
public:
	DownloadManager(void)
	{
		curl_global_init(CURL_GLOBAL_ALL);

		for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {

			threads[thread_idx].attach_signals(boost::bind(&DownloadManager::download_done_callback, this, _1), 
											   boost::bind(&DownloadManager::download_progress_callback, this), 
											   thread_idx);
		}
	}

	~DownloadManager(void)
	{
		curl_global_cleanup();
	}

	void download(const boost::gregorian::date& date)
	{
		if (!start_download(date)) {
			download_queue.push(date);
		}
	}

	bool start_download(const boost::gregorian::date &date)
	{
		int thread_idx;
		for (thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
			if (!threads[thread_idx].is_running()) {
				break;
			}
		}

		if (thread_idx == thread_max_count) {
			// No free thread
			return false;
		}

		std::string url = make_url(date);
		threads[thread_idx].start_download(url);
		active_thread_count++;
		return true;
	}

	void join(void)
	{
		while (active_thread_count > 0) {
			boost::posix_time::seconds seconds(1);
			boost::this_thread::sleep(seconds);
		}
	}

	void download_done_callback(int thread_idx)
	{
		printf("\nDownload done %d\n", thread_idx);
		active_thread_count--;
		if (download_queue.size() > 0) {
			if (start_download(download_queue.front())) {
				download_queue.pop();
			}
		}
	}

	void download_progress_callback(void)
	{
		bool first = true;
		printf("\33[2K\r");
		for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
			if (threads[thread_idx].is_running()) {
				if (!first) {
					printf("  ");
				} else {
					first = false;
				}

				printf("Progress(%d) % 3.1f MB", thread_idx, threads[thread_idx].get_progress());
			}
		}
		fflush(stdout);
	}


private:
	static const int thread_max_count = 2;
	DownloadManagerThread threads[thread_max_count];

	int active_thread_count = 0;

	std::queue<boost::gregorian::date> download_queue;

	std::string make_url(boost::gregorian::date date)
	{
		std::stringstream url;
		url.imbue(std::locale(std::cout.getloc(), new boost::date_time::date_facet < boost::gregorian::date, char>("%Y%m%d")));
		url << "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/" << date << ".csv.gz";
		return url.str();
	}
};

int main()
{
	DownloadManager download_manager;

	download_manager.download(boost::gregorian::date(2017, 05, 21));
	download_manager.download(boost::gregorian::date(2017, 05, 22));
	download_manager.download(boost::gregorian::date(2017, 05, 23));
	download_manager.download(boost::gregorian::date(2017, 05, 25));

	download_manager.join();

	return 0;
}
