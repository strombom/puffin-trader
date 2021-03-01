#include "pch.h"
#include "IE_Events.h"


IE_Event::IE_Event(time_point_ms timestamp, float price, float price_max, float price_min, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count) :
    timestamp(timestamp), price(price), price_max(price_max), price_min(price_min), delta(delta), delta_top(delta_top), delta_bot(delta_bot), duration(duration), volume(volume), trade_count(trade_count)
{

}

void IE_Events::append(time_point_ms timestamp, float price, float price_max, float price_min, float delta, float delta_top, float delta_bot, std::chrono::milliseconds duration, float volume, int trade_count)
{
    events.push_back({ timestamp, price, price_max, price_min, delta, delta_top, delta_bot, duration, volume, trade_count });
}
