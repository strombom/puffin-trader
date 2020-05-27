#include <cstdio>

#include "BitmexWebSocket.h"



int main()
{
    auto bitmex_web_socket = BitmexWebSocket{};
    
    bitmex_web_socket.start();

    printf("hello from BitmexDeltaServer!\n");
    return 0;
}
