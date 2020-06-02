#include "TickData.h"

#include <iostream>


TickData::TickData(void)
{
    for (auto &&symbol: Bitmex::symbols) {
        auto tick_array = std::make_unique<std::array<Tick, Bitmex::buffer_size>>();
        ticks.insert(std::make_pair(symbol, std::move(tick_array)));
        buffer_next_idx.insert(std::make_pair(symbol, 0));
        buffer_count.insert(std::make_pair(symbol, 0));
    }
}

std::shared_ptr<TickData> TickData::create(void)
{
    return std::make_shared<TickData>();
}

void TickData::append(const std::string& symbol, time_point_ms timestamp, float price, float volume, bool buy)
{
    auto slock = std::scoped_lock{ tick_data_mutex };

    if (ticks.count(symbol) == 0) {
        // Symbol does not exist in database
        return;
    }

    const auto idx = buffer_next_idx.at(symbol) % Bitmex::buffer_size;
    const auto tick = &(*ticks.at(symbol))[idx];

    tick->timestamp_ms = timestamp.time_since_epoch().count();
    tick->price = price;
    tick->volume = volume;
    tick->buy = buy;

    buffer_next_idx.at(symbol) = (buffer_next_idx.at(symbol) + 1) % Bitmex::buffer_size;
    buffer_count.at(symbol) = std::min(buffer_count.at(symbol) + 1, Bitmex::buffer_size);
    
    std::cout << "append " <<
        "sym(" << symbol << ") " <<
        "ts(" << DateTime::to_string(timestamp) << ") " <<
        "p(" << price << ") " <<
        "v(" << volume << ") " <<
        "b(" << buy << ") " <<
        "idx(" << buffer_next_idx.at(symbol) << ") " <<
        "cnt(" << buffer_count.at(symbol) << ") " <<
        std::endl;
    
}

std::unique_ptr<std::vector<Tick>> TickData::get(const std::string& symbol, time_point_ms timestamp)
{
    auto slock = std::scoped_lock{ tick_data_mutex };

    if (buffer_count.at(symbol) < 2) {
        return std::make_unique<std::vector<Tick>>();
    }

    //timestamp = ticks.at(symbol).get()->at((Bitmex::buffer_size + buffer_next_idx.at(symbol) - buffer_count.at(symbol) / 2) % Bitmex::buffer_size).timestamp();
    const auto timestamp_ms = timestamp.time_since_epoch().count();

    const auto last_idx = (Bitmex::buffer_size + buffer_next_idx.at(symbol) - 1) % Bitmex::buffer_size;
    const auto first_idx = buffer_count.at(symbol) < Bitmex::buffer_size ? 0 : buffer_next_idx.at(symbol);
    const auto read_ticks = ticks.at(symbol).get();
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
    auto result = std::vector<Tick>{};
    if (start_idx != -1) {
        idx = start_idx;
        while (true) {
            result.push_back(read_ticks->at(idx));
            if (idx == last_idx) {
                break;
            }
            idx = (Bitmex::buffer_size + idx + 1) % Bitmex::buffer_size;
        }
    }

    return std::make_unique<std::vector<Tick>>(result);
}