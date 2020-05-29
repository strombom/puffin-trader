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

    tick->timestamp = timestamp;
    tick->price = price;
    tick->volume = volume;
    tick->buy = buy;

    buffer_next_idx.at(symbol) = (buffer_next_idx.at(symbol) + 1) % Bitmex::buffer_size;
    buffer_count.at(symbol) = std::min(buffer_count.at(symbol) + 1, Bitmex::buffer_size);
    /*
    std::cout << "append " <<
        "sym(" << symbol << ") " <<
        "ts(" << DateTime::to_string(timestamp) << ") " <<
        "p(" << price << ") " <<
        "v(" << volume << ") " <<
        "b(" << buy << ") " <<
        "idx(" << buffer_idx.at(symbol) << ") " <<
        "cnt(" << buffer_count.at(symbol) << ") " <<
        std::endl;
    */
}

std::unique_ptr<std::vector<Tick>> TickData::get(const std::string& symbol, time_point_ms timestamp)
{
    auto slock = std::scoped_lock{ tick_data_mutex };

    const auto first_idx = (buffer_next_idx.at(symbol) - buffer_count.at(symbol)) % Bitmex::buffer_size;
    const auto last_idx = (buffer_next_idx.at(symbol) - 1) % Bitmex::buffer_size;
    const auto read_ticks = *ticks.at(symbol);
    auto start_idx = -1;

    // Find timestamp
    for (auto idx = last_idx; idx != (first_idx - 1) % Bitmex::buffer_size; idx = (idx - 1) % Bitmex::buffer_size) {
        if (read_ticks[idx].timestamp < timestamp) {
            start_idx = (idx + 1) % Bitmex::buffer_size;
            break;
        }
    }

    // Copy data
    auto result = std::vector<Tick>{};
    if (start_idx != -1) {
        for (auto idx = start_idx; idx != (last_idx + 1) % Bitmex::buffer_size; idx = (idx + 1) % Bitmex::buffer_size) {
            result.push_back(read_ticks[idx]);
        }
    }

    return std::make_unique<std::vector<Tick>>(result);
}