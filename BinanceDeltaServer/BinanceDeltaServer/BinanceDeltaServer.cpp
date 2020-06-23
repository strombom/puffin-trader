
#include <cstdio>

#include "BinanceWebSocket.h"
#include "TickData.h"
#include "Server.h"


int main()
{
    std::cout << "BinanceDeltaServer: Started" << std::endl;

    auto tick_data = TickData::create();
    auto server = Server{ tick_data };
    auto binance_web_socket = BinanceWebSocket{ tick_data };
    binance_web_socket.start();

    //getchar(); 
    //server.test();

    while (true) {
        auto command = std::string{};
        std::cin >> command;
        if (command.compare("q") == 0) {
            break;
        }
    }

    binance_web_socket.shutdown();

    std::cout << "BinanceDeltaServer: Shut down" << std::endl;
    return 0;
}
