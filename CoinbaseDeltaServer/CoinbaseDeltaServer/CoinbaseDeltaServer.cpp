
#include <cstdio>

#include "CoinbaseTick.h"
#include "TickData.h"
#include "Server.h"


int main()
{
    std::cout << "CoinbaseDeltaServer: Started" << std::endl;

    auto tick_data = TickData::create();
    auto server = Server{ tick_data };
    auto coinbase_tick = CoinbaseTick{ tick_data };
    coinbase_tick.start();

    while (true) {
        auto command = std::string{};
        std::cin >> command;
        if (command.compare("q") == 0) {
            break;
        }
    }

    coinbase_tick.shutdown();

    std::cout << "CoinbaseDeltaServer: Shut down" << std::endl;
    return 0;
}
