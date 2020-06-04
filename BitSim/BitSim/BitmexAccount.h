#pragma once
#include "pch.h"

#include "BitmexWebSocket.h"


class BitmexAccount
{
public:
    BitmexAccount(void);

    void start(void);
    void shutdown(void);

    double get_leverage(void);
    void limit_order(int contracts, double price);
    void market_order(int contracts);

private:
    sptrBitmexWebSocket bitmex_websocket;

};

using uptrBitmexAccount = std::unique_ptr<BitmexAccount>;
