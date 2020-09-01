
#include <cstdio>

#include "BitmexWebSocket.h"
#include "TickData.h"
#include "Server.h"


int main()
{
    std::cout << "BitmexOrderBookServer: Started" << std::endl;

    auto tick_data = TickData::create();
    auto server = Server{ tick_data };
    auto bitmex_web_socket = BitmexWebSocket{ tick_data };
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
