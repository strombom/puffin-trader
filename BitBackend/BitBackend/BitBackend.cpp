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

	//auto public_topics = std::vector<std::string>{};
	//public_topics.push_back("trade.BTCUSDT");
	//auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_public, false, public_topics);
	//bybit_public_websocket->start();

	auto private_topics = std::vector<std::string>{ "execution", "order", "position", "wallet" };

	auto bybit_private_websocket = std::make_shared<ByBitWebSocket>(ByBit::websocket::url_private, true, private_topics, portfolio);
	bybit_private_websocket->start();


	while (true) {
		std::this_thread::sleep_for(2s);
	}
	return 0;
}
