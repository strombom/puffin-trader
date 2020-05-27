#pragma once

#include <cpprest/ws_client.h>


class BitmexWebSocket
{
public:

    void start(void);

private:

    const char* base_url = "wss://www.bitmex.com/realtime?subscribe=trade:XBTUD";
};
