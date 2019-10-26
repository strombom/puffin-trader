// bitmex_get_daily.cpp
//

#pragma warning (disable : 26812)

#include <iostream>
#include <sstream>
#include <string>

#include "curl/curl.h"

size_t download_daily_callback(void* ptr, size_t size, size_t count, void* stream)
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
	if (download_count >= (int) 10e5) {
		download_count -= (int) 10e5;
		printf("\rDownloading %0.1f MB", download_count_tot / 10e6);
		fflush(stdout);
	}

	// Add data to download data stream
	((std::stringstream*) stream)->write((const char*)ptr, (std::streamsize) count);
	return count;
}

void download_daily(const std::string &url) {
	CURL* curl;
	std::stringstream download_data;

	curl_global_init(CURL_GLOBAL_ALL);

	curl = curl_easy_init();
	if (curl) {
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, FALSE);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &download_data);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, download_daily_callback);
		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

#ifdef SKIP_PEER_VERIFICATION
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
#endif

#ifdef SKIP_HOSTNAME_VERIFICATION
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
#endif

		printf("Downloading 0 MB");
		fflush(stdout);

		CURLcode res = curl_easy_perform(curl);

		if (res != CURLE_OK) {
			printf(", failed.\n");
			//fprintf(stderr, "curl_easy_perform() failed: %s\n",
			//	curl_easy_strerror(res));
		} else {
			printf(", OK.\n");
		}

		//download_data.seekg(0, std::stringstream::end);
		//unsigned int length = (int) download_data.tellg();
		

		curl_easy_cleanup(curl);
	}
	curl_global_cleanup();
}

int main()
{
	download_daily("https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/20171121.csv.gz");
}
