#include "OrderBookData.h"

#include <iostream>


OrderBookData::OrderBookData(void)
{
    for (auto &&symbol: Bitmex::symbols) {
        auto order_book_tick_array = std::make_unique<std::array<OrderBook, Bitmex::buffer_size>>();
        order_book_ticks.insert(std::make_pair(symbol, std::move(order_book_tick_array)));
        buffer_next_idx.insert(std::make_pair(symbol, 0));
        buffer_count.insert(std::make_pair(symbol, 0));
    }
}

std::shared_ptr<OrderBookData> OrderBookData::create(void)
{
    return std::make_shared<OrderBookData>();
}

void OrderBookData::append(const std::string& symbol, time_point_ms timestamp, float bid_price, float bid_volume, float ask_price, float ask_volume)
{
    auto slock = std::scoped_lock{ tick_data_mutex };

    if (order_book_ticks.count(symbol) == 0) {
        // Symbol does not exist in database
        return;
    }

    const auto idx = buffer_next_idx.at(symbol) % Bitmex::buffer_size;
    const auto order_book_tick = &(*order_book_ticks.at(symbol))[idx];

    order_book_tick->timestamp_ms = timestamp.time_since_epoch().count();
    order_book_tick->bid_price = bid_price;
    order_book_tick->bid_volume = bid_volume;
    order_book_tick->ask_price = ask_price;
    order_book_tick->ask_volume = ask_volume;

    buffer_next_idx.at(symbol) = (buffer_next_idx.at(symbol) + 1) % Bitmex::buffer_size;
    buffer_count.at(symbol) = std::min(buffer_count.at(symbol) + 1, Bitmex::buffer_size);
    
    std::cout << "append " <<
        "sym(" << symbol << ") " <<
        "ts(" << DateTime::to_string(timestamp) << ") " <<
        "bp(" << bid_price << ") " <<
        "bv(" << bid_volume << ") " <<
        "ap(" << ask_price << ") " <<
        "av(" << ask_volume << ") " <<
        "idx(" << buffer_next_idx.at(symbol) << ") " <<
        "cnt(" << buffer_count.at(symbol) << ") " <<
        std::endl;
    
}

std::unique_ptr<std::vector<OrderBook>> OrderBookData::get(const std::string& symbol, time_point_ms timestamp, int max_rows)
{
    auto slock = std::scoped_lock{ tick_data_mutex };

    if (buffer_count.at(symbol) < 2) {
        return std::make_unique<std::vector<OrderBook>>();
    }

    //timestamp = ticks.at(symbol).get()->at((Bitmex::buffer_size + buffer_next_idx.at(symbol) - buffer_count.at(symbol) / 2) % Bitmex::buffer_size).timestamp();
    const auto timestamp_ms = timestamp.time_since_epoch().count();

    const auto last_idx = (Bitmex::buffer_size + buffer_next_idx.at(symbol) - 1) % Bitmex::buffer_size;
    const auto first_idx = buffer_count.at(symbol) < Bitmex::buffer_size ? 0 : buffer_next_idx.at(symbol);
    const auto read_ticks = order_book_ticks.at(symbol).get();
    auto start_idx = -1;

    std::cout << "timestamp count:" << timestamp.time_since_epoch().count() << std::endl;
    std::cout << "ticks timestamp_ms:" << read_ticks->at(first_idx).timestamp_ms << std::endl;
    std::cout << "ticks timestamp() count:" << read_ticks->at(first_idx).timestamp().time_since_epoch().count() << std::endl;

    std::cout << "first " << DateTime::to_string(read_ticks->at(first_idx).timestamp()) << std::endl;
    std::cout << "last_idx " << DateTime::to_string(read_ticks->at(last_idx).timestamp()) << std::endl;
    std::cout << "search " << DateTime::to_string(timestamp) << std::endl;

    // Find timestamp
    auto idx = last_idx;
    while (true) {
        //std::cout << "search " << idx << std::endl;
        if (read_ticks->at(idx).timestamp_ms < timestamp_ms) {
            if (idx != last_idx) {
                start_idx = (idx + 1) % Bitmex::buffer_size;
            }
            break;
        }
        if (idx == first_idx) {
            break;
        }
        idx = (Bitmex::buffer_size + idx - 1) % Bitmex::buffer_size;
    }

    // Copy data
    auto result = std::vector<OrderBook>{};
    if (start_idx != -1) {
        idx = start_idx;
        while (true) {
            result.push_back(read_ticks->at(idx));
            if (idx == last_idx || result.size() == max_rows) {
                break;
            }
            idx = (Bitmex::buffer_size + idx + 1) % Bitmex::buffer_size;
        }
    }

    return std::make_unique<std::vector<OrderBook>>(result);
}