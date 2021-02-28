#include "pch.h"
#include "IE_Events.h"


IE_Event::IE_Event(time_point_ms timestamp, float price, float delta, std::chrono::milliseconds duration, float volume, float spread, int trade_count) :
    timestamp(timestamp), price(price), delta(delta),duration(duration), volume(volume), spread(spread), trade_count(trade_count)
{

}

void IE_Events::append(time_point_ms timestamp, float price, float delta, std::chrono::milliseconds duration, float volume, float spread, int trade_count)
{
    events.push_back({ timestamp, price, delta, duration, volume, spread, trade_count });
}
