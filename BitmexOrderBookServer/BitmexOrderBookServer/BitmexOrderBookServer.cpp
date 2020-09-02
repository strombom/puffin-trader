
#include <cstdio>

#include "BitmexWebSocket.h"
#include "OrderBookData.h"
#include "Server.h"


int main()
{
    std::cout << "BitmexOrderBookServer: Started" << std::endl;

    auto order_book_data = OrderBookData::create();
    auto server = Server{ order_book_data };
    auto bitmex_web_socket = BitmexWebSocket{ order_book_data };
    bitmex_web_socket.start();

    while (true) {
        auto command = std::string{};
        std::cin >> command;
        if (command.compare("q") == 0) {
            break;
        }
    }

    bitmex_web_socket.shutdown();

    std::cout << "BitmexOrderBookServer: Shut down" << std::endl;
    return 0;
}
