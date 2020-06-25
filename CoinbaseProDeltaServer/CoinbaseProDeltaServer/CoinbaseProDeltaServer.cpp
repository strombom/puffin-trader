
#include "CoinbaseProWebSocket.h"
#include "TickData.h"
#include "Server.h"

#include <cstdio>
#include <iostream>


int main()
{
    std::cout << "CoinbaseProDeltaServer: Started" << std::endl;

    auto tick_data = TickData::create();
    auto server = Server{ tick_data };
    auto coinbase_websocket = CoinbaseProWebSocket{ tick_data };
    coinbase_websocket.start();

    while (true) {
        auto command = std::string{};
        std::cin >> command;
        if (command.compare("q") == 0) {
            break;
        }
    }

    coinbase_websocket.shutdown();

    std::cout << "CoinbaseProDeltaServer: Shut down" << std::endl;
    return 0;
}
