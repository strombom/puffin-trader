#include "pch.h"
#include "BitmexAccount.h"

BitmexAccount::BitmexAccount(void)
{
    bitmex_websocket = std::make_shared<BitmexWebSocket>();
}

void BitmexAccount::start(void)
{
    bitmex_websocket->start();
}

void BitmexAccount::shutdown(void)
{
    bitmex_websocket->shutdown();
}

double BitmexAccount::get_leverage(void)
{
    return 0.0;
}

void BitmexAccount::limit_order(int contracts, double price)
{

}

void BitmexAccount::market_order(int contracts)
{

}
