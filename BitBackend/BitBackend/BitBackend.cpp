// BitBackend.cpp : Defines the entry point for the application.
//

#include "ByBitWebsocket.h"

#include <iostream>
#include <boost/beast/websocket.hpp>


int main()
{
	auto bybit_websocket = ByBitWebSocket{};
	bybit_websocket.start();

	std::cout << "Hello CMake." << std::endl;
	return 0;
}
