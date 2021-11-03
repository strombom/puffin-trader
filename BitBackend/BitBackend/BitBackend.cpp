#include "precompiled_headers.h"

#include "ByBitWebsocket.h"
#include "ByBitConfig.h"
#include "Portfolio.h"

#include "BitLib/DateTime.h"

using namespace std::chrono_literals;

int main()
{
	/*
	
	auto a = Uuid{ "123e4567-e89b-12d3-a456-426614174000" };
	auto b = Uuid{ "123e4567-e89b-12d3-a456-426614174000" };
	auto c = Uuid{ "223e4567-e89b-12d3-a456-426614174000" };

	if (a == b) {
		printf("a==b\n");
	}

	if (a == c) {
		printf("a==c\n");
	}

	return 0;
	*/

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
