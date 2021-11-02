#include "precompiled_headers.h"

#include "ByBitWebsocket.h"
#include "ByBitConfig.h"

using namespace std::chrono_literals;

int main()
{
	auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_public, false);
	bybit_public_websocket->start();

	while (true) {
		std::cout << "Hello CMake." << std::endl;
		std::this_thread::sleep_for(2s);
	}
	return 0;
}
