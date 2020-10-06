#pragma once
#include "pch.h"

#include "BitLib/AggTicks.h"

using namespace std::chrono_literals;


enum class PD_Direction {
    up,
    down
};

class PD_OrderBookBuffer {
public:
    PD_OrderBookBuffer(void);

    void update(time_point_ms timestamp, float price_low, float price_high);

    std::tuple<float, float> get_price(time_point_ms timestamp);

private:
    static constexpr int size = 1000;
    std::array<time_point_ms, size> timestamps;
    std::array<float, size> prices_low;
    std::array<float, size> prices_high;
    int length;
    int next_idx;

    float order_book_bottom;
};

class PD_OrderBook {
public:
    PD_OrderBook(void);
    PD_OrderBook(time_point_ms timestamp, float price_low, float price_high);

    bool update(time_point_ms timestamp, float price_low, float price_high, PD_Direction direction);

    PD_OrderBookBuffer buffer;
private:
};

class PD_Event
{
public:
    PD_Event(PD_Direction direction, time_point_ms timestamp_delta, time_point_ms timestamp_overshoot, float price_delta, float price_overshoot, size_t agg_tick_idx_delta, size_t agg_tick_idx_overshoot) :
        direction(direction),
        timestamp_delta(timestamp_delta),
        timestamp_overshoot(timestamp_overshoot),
        price_delta(price_delta),
        price_overshoot(price_overshoot),
        agg_tick_idx_delta(agg_tick_idx_delta),
        agg_tick_idx_overshoot(agg_tick_idx_overshoot)
    {}

    friend std::ostream& operator<<(std::ostream& stream, const AggTick& agg_tick);

    PD_Direction direction;
    time_point_ms timestamp_delta;
    time_point_ms timestamp_overshoot;
    float price_delta;
    float price_overshoot;
    size_t agg_tick_idx_delta;
    size_t agg_tick_idx_overshoot;
};

using sptrPD_Event = std::shared_ptr<PD_Event>;


class PD_Events
{
public:
    PD_Events(double delta);
    PD_Events(double delta, sptrAggTicks agg_ticks);

    std::vector<PD_Event> events;
    //std::vector<PD_Event> events_offset;

    friend std::ostream& operator<<(std::ostream& stream, const PD_Events& pd_events_data);

    void save(const std::string& filename_path) const;

    void plot_events(sptrAggTicks agg_ticks, const std::string& filename);

    sptrPD_Event update(sptrAggTick agg_tick);

    /*
    PD_Events(const Tick& first_tick);
    PD_Events(time_point_ms timestamp, const Interval& first_interval);

    PD_Events(sptrTicks ticks);
    PD_Events(sptrIntervals intervals);

    sptrPD_Event step(const Interval& intervals);

    void plot_events(sptrIntervals intervals);
    */

private:
    size_t event_idx;

    const std::chrono::milliseconds offset = 200ms;
    const double delta;

    PD_Direction last_direction;
    PD_OrderBook order_book;
    float price_min;
    float price_max;

    //std::vector<PD_Event> events_offset;
    //std::vector<PD_Event> tick_prices;
    //std::vector<PD_Event> order_book_top;
    //std::vector<PD_Event> order_book_bot;
};

using sptrPD_Events = std::shared_ptr<PD_Events>;
