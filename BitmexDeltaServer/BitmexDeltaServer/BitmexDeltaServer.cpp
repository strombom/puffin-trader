
#include <cstdio>

#include "BitmexWebSocket.h"
#include "TickData.h"
#include "Server.h"


int main()
{
    printf("BitmexDeltaServer: Started\n");

    auto tick_data = TickData::create();
    auto server = Server{ tick_data };
    auto bitmex_web_socket = BitmexWebSocket{ tick_data };
    bitmex_web_socket.start();

    getchar(); // Wait for Enter key

    bitmex_web_socket.shutdown();
    
    printf("BitmexDeltaServer: Shut down\n");
    return 0;
}
