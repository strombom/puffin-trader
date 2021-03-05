#include "pch.h"
#include "IE_Events.h"


IE_Event::IE_Event(time_point_ms timestamp, float price, float price_max, float price_min, float price_buy, float price_sell, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count) :
    timestamp(timestamp), price(price), price_max(price_max), price_min(price_min), price_buy(price_buy), price_sell(price_sell), delta(delta), delta_top(delta_top), delta_bot(delta_bot), duration(duration), volume(volume), trade_count(trade_count)
{

}

void IE_Events::append(time_point_ms timestamp, float price, float price_max, float price_min, float price_buy, float price_sell, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count)
{
    events.push_back({ timestamp, price, price_max, price_min, price_buy, price_sell, delta, delta_top, delta_bot, duration, volume, trade_count });
}
