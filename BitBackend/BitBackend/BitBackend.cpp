#include "precompiled_headers.h"

#include "BitLib/DateTime.h"
#include "ByBitWebsocket.h"
#include "OrderManager.h"
#include "ByBitConfig.h"
#include "OrderBook.h"
#include "Portfolio.h"
#include "ByBitRest.h"
#include "Symbols.h"

#include <filesystem>

using namespace std::chrono_literals;

int main()
{
	auto portfolio = std::make_shared<Portfolio>();
	auto order_books = makeOrderBooks(); // std::make_shared<OrderBooks>();
	auto order_manager = std::make_shared<OrderManager>(portfolio, order_books);

	auto bybit_rest = ByBitRest{ order_manager };
	//while (!bybit_rest.is_connected()) {
	//	std::this_thread::sleep_for(100ms);
	//}

	/*
	//auto order_book_updated_callback = order_manager.get_update_callback();

	auto public_topics = std::vector<std::string>{};
	for (const auto& symbol : symbols) {
		public_topics.push_back(std::string{ "orderBookL2_25." } + symbol.name.data());
	}
    auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::WebSocket::url_public, false, public_topics, order_manager);
	bybit_public_websocket->start();

	auto private_topics = std::vector<std::string>{ "execution", "order", "position", "wallet" };
	auto bybit_private_websocket = std::make_shared<ByBitWebSocket>(ByBit::WebSocket::url_private, true, private_topics, order_manager);
	bybit_private_websocket->start();

	//bybit_rest.place_order("BTCUSDT", 0.01, 51000.0);
	//bybit_rest.cancel_order("BTCUSDT", 3);

	*/

	for (const auto &symbol : symbols) {
		bybit_rest.get_position(symbol);
	}	

	bybit_rest.join();

	return 0;
}
