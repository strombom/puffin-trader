#include "pch.h"
#include "PD_Events.h"


using namespace std::chrono_literals;


PD_OrderBookBuffer::PD_OrderBookBuffer(void) :
    length(0),
    next_idx(0),
    order_book_bottom(0.0)
{
        
}

void PD_OrderBookBuffer::update(time_point_ms timestamp, double price)
{
    if (price > order_book_bottom + 0.5) {
        order_book_bottom = price - 0.5;
    }
    else if (price < order_book_bottom) {
        order_book_bottom = price;
    }

    timestamps[next_idx] = timestamp;
    prices[next_idx] = price;

    length = std::min(length + 1, size);
    next_idx = (next_idx + 1) % size;
}

std::tuple<double, double> PD_OrderBookBuffer::get_price(time_point_ms timestamp)
{
    auto count = 0;
    auto idx = (next_idx - 1 + size) % size;
    auto price_bot = std::numeric_limits<double>::max();
    auto price_top = std::numeric_limits<double>::min();
    while (count < length) {
        price_bot = std::min(price_bot, prices[idx]);
        price_top = std::max(price_top, prices[idx]);

        if (timestamps[idx] < timestamp) {
            break;
        }
        idx = (idx - 1 + size) % size;
        ++count;
    }
    if (price_top < order_book_bottom + 0.5) {
        price_top = order_book_bottom + 0.5;
    }
    else if (price_bot > order_book_bottom) {
        price_bot = order_book_bottom;
    }

    return std::make_tuple(price_bot, price_top);
}

PD_OrderBook::PD_OrderBook(time_point_ms timestamp, double price)
{
    buffer.update(timestamp, price);
}

bool PD_OrderBook::update(time_point_ms timestamp, double price, PD_Direction direction)
{
    const auto [price_bot, price_top] = buffer.get_price(timestamp - 1500ms);
    buffer.update(timestamp, price);
    if (direction == PD_Direction::down && price > price_top) {
        return true;
    }
    else if (direction == PD_Direction::up && price < price_bot) {
        return true;
    }
    return false;
}

PD_Events::PD_Events(sptrAggTicks agg_ticks)
{
    if (agg_ticks->agg_ticks.size() < 2) {
        return;
    }

    const auto offset = 200ms;
    auto last_direction = PD_Direction::up;
    auto order_book = PD_OrderBook{ agg_ticks->agg_ticks[0].timestamp, agg_ticks->agg_ticks[0].low };
    auto finding_offset = false;

    for (auto agg_tick_idx = 0; agg_tick_idx < agg_ticks->agg_ticks.size(); agg_tick_idx++) {
        const auto agg_tick = &agg_ticks->agg_ticks[agg_tick_idx];
        if (finding_offset) {
            if (agg_tick->timestamp >= events.back().timestamp + offset) {
                events_offset.push_back(PD_Event{ agg_tick->timestamp, agg_tick->low, last_direction, (size_t)agg_tick_idx });
                finding_offset = false;
            }
        }

        //auto event = step(agg_tick);
        const auto executed_low = order_book.update(agg_tick->timestamp, agg_tick->low, last_direction);
        const auto executed_high = order_book.update(agg_tick->timestamp, agg_tick->high, last_direction);
        if (executed_low || executed_high) {
            if (last_direction == PD_Direction::up) {
                last_direction = PD_Direction::down;
            }
            else {
                last_direction = PD_Direction::up;
            }

            auto execution_price = executed_low ? agg_tick->low : agg_tick->high;
            const auto event = PD_Event{ agg_tick->timestamp, execution_price, last_direction, (size_t)agg_tick_idx };
            if (finding_offset) {
                events_offset.push_back(event);
            }
            events.push_back(event);
            finding_offset = true;
        }
    }

    // If no offset was found for last event, remove it
    if (events_offset.size() > events.size()) {
        events_offset.pop_back();
    }

}

/*
PD_Events::PD_Events(const Tick& first_tick) :
    order_book(first_tick.timestamp, first_tick.price),
    last_direction(PD_Direction::up)
{

}

PD_Events::PD_Events(time_point_ms timestamp, const Interval& first_interval) :
    order_book(timestamp, first_interval.last_price),
    last_direction(PD_Direction::up)
{

}

PD_Events::PD_Events(sptrTicks ticks) :
    PD_Events(ticks->rows[0])
{
    if (ticks->rows.size() < 2) {
        return;
    }

    const auto offset = 200ms;
    order_book = PD_OrderBook{ ticks->rows[0].timestamp, ticks->rows[0].price };
    auto finding_offset = false;

    for (auto&& tick : ticks->rows) {
        if (finding_offset) {
            if (tick.timestamp >= events.back().timestamp + offset) {
                events_offset.push_back(PD_Event{ tick.timestamp, tick.price, last_direction });
                finding_offset = false;
            }
        }
        auto event = step(tick);

        if (event) {
            if (finding_offset) {
                events_offset.push_back(PD_Event{ tick.timestamp, tick.price, last_direction });
            }
            events.push_back(*event);
            finding_offset = true;
        }

        tick_prices.push_back(PD_Event{ tick.timestamp, tick.price, last_direction });

        const auto [price_bot, price_top] = order_book.buffer.get_price(tick.timestamp - 1500ms);
        order_book_bot.push_back(PD_Event{ tick.timestamp, price_bot, last_direction });
        order_book_top.push_back(PD_Event{ tick.timestamp, price_top, last_direction });
    }

    // If no offset was found for last event, remove it
    if (events_offset.size() > events.size()) {
        events_offset.pop_back();
    }
}

PD_Events::PD_Events(sptrIntervals intervals) :
    PD_Events(intervals->get_timestamp_start(), intervals->rows[0])
{

}

sptrPD_Event PD_Events::step(const Tick& tick)
{
    const auto executed = order_book.update(tick.timestamp, tick.price, last_direction);
    if (executed) {
        if (last_direction == PD_Direction::up) {
            last_direction = PD_Direction::down;
        }
        else {
            last_direction = PD_Direction::up;
        }
        return std::make_shared<PD_Event>(tick.timestamp, tick.price, last_direction);
    }

    return nullptr;
}

void PD_Events::plot_events(sptrIntervals intervals)
{
    auto event_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\events.csv" };
    for (auto&& event : events) {
        const auto interval_idx = (event.timestamp.time_since_epoch() - BitSim::timestamp_start.time_since_epoch()) / BitSim::interval;
        event_file << interval_idx << ",";
        event_file << event.price << ",";
        event_file << (int)event.direction << '\n';
    }
    event_file.close();

    auto event_offset_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\events_offset.csv" };
    for (auto&& event : events_offset) {
        const auto interval_idx = (event.timestamp.time_since_epoch() - BitSim::timestamp_start.time_since_epoch()) / BitSim::interval;
        event_offset_file << interval_idx << ",";
        event_offset_file << event.price << ",";
        event_offset_file << (int)event.direction << '\n';
    }
    event_offset_file.close();

    auto interval_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\intervals.csv" };
    for (auto&& interval : intervals->rows) {
        interval_file << interval.last_price << '\n';
    }
    interval_file.close();

    auto tick_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\ticks.csv" };
    for (auto&& event : tick_prices) {
        const auto diff_ms = (event.timestamp.time_since_epoch() - BitSim::timestamp_start.time_since_epoch()).count();
        const auto interval_idx = ((double)diff_ms) / BitSim::interval.count();
        tick_file << interval_idx << ",";
        tick_file << event.price << '\n';
    }
    event_file.close();

    auto obt_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\orderbook_top.csv" };
    for (auto&& event : order_book_top) {
        const auto interval_idx = (event.timestamp.time_since_epoch() - BitSim::timestamp_start.time_since_epoch()) / BitSim::interval;
        obt_file << interval_idx << ",";
        obt_file << event.price << '\n';
    }
    obt_file.close();

    auto obb_file = std::ofstream{ std::string{ BitSim::tmp_path } + "\\pd_events\\orderbook_bot.csv" };
    for (auto&& event : order_book_bot) {
        const auto interval_idx = (event.timestamp.time_since_epoch() - BitSim::timestamp_start.time_since_epoch()) / BitSim::interval;
        obb_file << interval_idx << ",";
        obb_file << event.price << '\n';
    }
    obb_file.close();
}
*/
