
#include <cstdio>

#include "BitmexWebSocket.h"


int main()
{
    printf("BitmexDeltaServer: Started\n");
    
    auto bitmex_web_socket = BitmexWebSocket{};    
    bitmex_web_socket.start();

    getchar(); // Wait for Enter key

    bitmex_web_socket.shutdown();
    
    printf("BitmexDeltaServer: Shut down\n");
    return 0;
}
