#include "precompiled_headers.h"

#include "ByBitWebsocket.h"
#include "ByBitConfig.h"
#include "Portfolio.h"

#include "BitLib/DateTime.h"
#include <filesystem>
using namespace std::chrono_literals;

int main()
{
	auto portfolio = std::make_shared<Portfolio>();

	auto public_topics = std::vector<std::string>{};
	for (const auto& symbol : symbols) {
		const auto topic = std::string{ "trade." } + symbol.name.data();
		public_topics.push_back(topic);
	}
    auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_public, false, public_topics, portfolio);
	bybit_public_websocket->start();

	auto private_topics = std::vector<std::string>{ "execution", "order", "position", "wallet" };
	auto bybit_private_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_private, true, private_topics, portfolio);
	//bybit_private_websocket->start();

	while (true) {
		std::this_thread::sleep_for(30s);
		//bybit_private_websocket->send_heartbeat();
		bybit_public_websocket->send_heartbeat();
	}
	return 0;
}
