// bitmex_get_daily.cpp
//

#pragma warning (disable : 26812)

#include <iostream>
#include <sstream>
#include <string>

#include "boost/date_time/gregorian/gregorian.hpp"
#include "curl/curl.h"

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

void download_file(const std::string& url) {
	std::stringstream download_data;

	curl_global_init(CURL_GLOBAL_ALL);
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
	curl_global_cleanup();


	download_data.seekg(0, std::stringstream::end);
	unsigned int length = (int) download_data.tellg();
	printf("file size %d\n", length);
}

std::string make_url(boost::gregorian::date date)
{	
	std::locale loc(std::cout.getloc(), new boost::date_time::date_facet < boost::gregorian::date, char>("%Y%m%d"));
	std::stringstream url;
	url.imbue(loc);
	url << "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/" << date << ".csv.gz";
	return url.str();
}

int main()
{
	boost::gregorian::date date {2017, 11, 21};
	std::string url = make_url(date);
	printf("Downloading URL: %s\n", url.c_str());
	download_file(url);
}
