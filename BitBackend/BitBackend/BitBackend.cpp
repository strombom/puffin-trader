#include "precompiled_headers.h"

#include "ByBitWebsocket.h"
#include "OrderManager.h"
#include "ByBitConfig.h"
#include "Portfolio.h"
#include "ByBitRest.h"

#include "BitLib/DateTime.h"
#include <filesystem>
using namespace std::chrono_literals;

int main()
{
	auto bybit_rest = ByBitRest();

	while (!bybit_rest.is_connected()) {
		std::this_thread::sleep_for(100ms);
	}

	bybit_rest.place_order("BTCUSDT", 0.001, 50000.0);

	bybit_rest.join();

	return 1;
	auto portfolio = std::make_shared<Portfolio>();

	auto public_topics = std::vector<std::string>{};
	for (const auto& symbol : symbols) {
		//public_topics.push_back(std::string{ "trade." } + symbol.name.data());
		public_topics.push_back(std::string{ "orderBookL2_25." } + symbol.name.data());
	}
    auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_public, false, public_topics, portfolio);
	bybit_public_websocket->start();

	auto private_topics = std::vector<std::string>{ "execution", "order", "position", "wallet" };
	auto bybit_private_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_private, true, private_topics, portfolio);
	//bybit_private_websocket->start();

	auto order_manager = OrderManager{ portfolio };

	while (true) {
		std::this_thread::sleep_for(1s);
	}
	return 0;
}
