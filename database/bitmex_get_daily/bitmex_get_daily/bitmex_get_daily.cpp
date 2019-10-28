// bitmex_get_daily.cpp
//

#pragma warning (disable : 26812)
#pragma warning (disable : 26444)

#include <iostream>
#include <sstream>
#include <string>

#include "curl/curl.h"
#include "boost/bind.hpp"
#include "boost/thread.hpp"
#include "boost/lockfree/queue.hpp"
#include "boost/date_time/gregorian/gregorian.hpp"

//using namespace boost;

size_t download_file_callback(void* ptr, size_t size, size_t count, void* stream)
{
	static void* stream_id = 0;
	static int download_count = 0;
	static int download_count_tot = 0;

	if (stream_id != stream) {
		download_count = 0;
		stream_id = stream;
	}

	download_count += (int)count;
	download_count_tot += (int)count;
	if (download_count >= (int)10e5) {
		download_count -= (int)10e5;
		// Display progress
		printf("\rDownloading %0.1f MB", download_count_tot / 10e6);
		fflush(stdout);
	}

	// Add data to download data stream
	((std::stringstream*) stream)->write((const char*)ptr, (std::streamsize) count);
	return count;
}

class DownloadManagerThread {
public:

	void download(const std::string& url)
	{
		running = true;
		thread = new boost::thread(&DownloadManagerThread::download_file, this, url);
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

private:
	bool running = false;
	boost::thread* thread;

	void download_file(const std::string& url)
	{
		std::stringstream download_data;

		CURL* curl = curl_easy_init();
		if (curl) {
			curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, &download_data);
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_file_callback);
			curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

			printf("Downloading 0 MB");
			fflush(stdout);

			CURLcode res = curl_easy_perform(curl);

			if (res != CURLE_OK) {
				printf(", failed.\n");
				printf("CURL error: %s\n", curl_easy_strerror(res));
				download_data.clear();
			}
			else {
				printf(", OK.\n");
			}

			curl_easy_cleanup(curl);
		}

		download_data.seekg(0, std::stringstream::end);
		unsigned int length = (int)download_data.tellg();
		printf("file size %d\n", length);

		printf("pos1 %d\n", (int)thread);
		delete thread;
		printf("pos2 %d\n", (int)thread);
		thread = NULL;
		printf("pos3 %d\n", (int)thread);


	}

};

class DownloadManager {
public:
	DownloadManager(void)
	{
		curl_global_init(CURL_GLOBAL_ALL);

		//manager_thread = new boost::thread(&DownloadManager::manager, this);
	}

	~DownloadManager(void)
	{
		curl_global_cleanup();
	}

	bool download(const boost::gregorian::date &date)
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
		printf("Downloading URL: %s\n", make_url(date).c_str());

		active_thread_count++;
		threads[thread_idx].download(url);

		return true;
	}

	void join(void)
	{
		while (active_thread_count > 0) {
			boost::posix_time::seconds seconds(1);
			boost::this_thread::sleep(seconds);
		}
		//for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
		//	threads[thread_idx].join();
		//}
	}

private:
	static const int thread_max_count = 5;
	DownloadManagerThread threads[thread_max_count];

	boost::thread* manager_thread;
	int active_thread_count = 0;

	void manager(void)
	{

	}

	std::string make_url(boost::gregorian::date date)
	{
		std::stringstream url;
		url.imbue(std::locale(std::cout.getloc(), new boost::date_time::date_facet < boost::gregorian::date, char>("%Y%m%d")));
		url << "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/" << date << ".csv.gz";
		return url.str();
	}

	/*
		for (int thread_idx = 0; thread_idx < thread_max_count; thread_idx++) {
			curl[thread_idx] = NULL;
			thread_running[thread_idx] = false;
		}

	CURL* curl[thread_max_count];
	bool thread_running[thread_max_count];
	*/

	/*
	printf("Starting thread 1\n");
	boost::thread downloader_thread1(downloader, boost::gregorian::date(2017, 11, 21));
	printf("Starting thread 2\n");
	boost::thread downloader_thread2(downloader, boost::gregorian::date(2017, 11, 22));
	printf("Starting thread 3\n");
	boost::thread downloader_thread3(downloader, boost::gregorian::date(2017, 11, 23));

	printf("Waiting for thread 1\n");
	downloader_thread1.join();
	downloader_thread2.join();
	downloader_thread3.join();

	printf("Done threads\n");
	*/
};

int main()
{
	DownloadManager download_manager;

	download_manager.download(boost::gregorian::date(2017, 11, 21));
	//download_manager.download(boost::gregorian::date(2017, 11, 22));
	//download_manager.download(boost::gregorian::date(2017, 11, 23));

	download_manager.join();

	return 0;
}
