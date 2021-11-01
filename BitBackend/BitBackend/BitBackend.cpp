// BitBackend.cpp : Defines the entry point for the application.
//

#include "ByBitWebsocket.h"

#include <boost/beast/websocket.hpp>
#include <iostream>
#include <thread>
#include <chrono>
using namespace std::chrono_literals;

int main()
{
	auto bybit_websocket = std::make_shared<ByBitWebSocket>();
	bybit_websocket->start();

	while (true) {
		std::cout << "Hello CMake." << std::endl;
		std::this_thread::sleep_for(2s);
	}
	return 0;
}
