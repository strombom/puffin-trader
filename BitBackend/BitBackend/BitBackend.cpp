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

	auto public_topics = std::vector<std::string>{};
	for (const auto& symbol : symbols) {
		public_topics.push_back(std::string{ "orderBookL2_25." } + symbol.name.data());
	}
    auto bybit_public_websocket = std::make_shared<ByBitWebSocket>(ByBit::WebSocket::url_public, false, public_topics, order_manager);
	bybit_public_websocket->start();

	auto private_topics = std::vector<std::string>{ "execution", "order", "position" }; // , "wallet"
	auto bybit_private_websocket = std::make_shared<ByBitWebSocket>(ByBit::WebSocket::url_private, true, private_topics, order_manager);
	bybit_private_websocket->start();

	auto bybit_rest = ByBitRest{ order_manager };
	for (const auto &symbol : symbols) {
		bybit_rest.cancel_all_orders(symbol);
		bybit_rest.get_position(symbol);
	}
	std::this_thread::sleep_for(4s);
	//bybit_rest.place_order(string_to_symbol("BTCUSDT"), 0.01, 51000.0);

	bybit_rest.join();

	return 0;
}
