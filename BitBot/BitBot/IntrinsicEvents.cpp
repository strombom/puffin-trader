#include "pch.h"

#include "IntrinsicEvents.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


IntrinsicEvents::IntrinsicEvents(std::string symbol) : symbol(symbol)
{
    load();
}

std::ostream& operator<<(std::ostream& stream, const IntrinsicEvent& row)
{
    stream.write(reinterpret_cast<const char*>(&row.timestamp), sizeof(row.timestamp));
    stream.write(reinterpret_cast<const char*>(&row.price), sizeof(row.price));

    return stream;
}

std::istream& operator>>(std::istream& stream, IntrinsicEvent& row)
{
    stream.read(reinterpret_cast <char*> (&row.timestamp), sizeof(row.timestamp));
    stream.read(reinterpret_cast <char*> (&row.price), sizeof(row.price));

    return stream;
}

std::ostream& operator<<(std::ostream& stream, const IntrinsicEvents& intrinsic_events)
{
    for (auto&& row : intrinsic_events.events) {
        stream << row;
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, IntrinsicEvents& intrinsic_events)
{
    auto intrinsic_event = IntrinsicEvent{};
    while (stream >> intrinsic_event) {
        intrinsic_events.events.push_back(intrinsic_event);
    }

    return stream;
}

void IntrinsicEvents::load(void)
{
    const auto file_path = std::string{ BitBot::path } + "\\intrinsic_events\\" + symbol + ".dat";
    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        auto intrinsic_event = IntrinsicEvent{};
        while (data_file >> intrinsic_event) {
            events.push_back(intrinsic_event);
        }
        data_file.close();
    }
}

void IntrinsicEvents::save(void) const
{
    auto file_path = std::string{ BitBot::path } + "\\intrinsic_events";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };
    data_file << *this;
    data_file.close();
}

std::vector<double> IntrinsicEventRunner::step(double price)
{
    if (!initialized) {
        current_price = price;
        previous_price = price;
        ie_start_price = price;
        ie_max_price = price;
        ie_min_price = price;
        initialized = true;
    }
            
    auto ie_prices = std::vector<double>{};

    if (price > current_price) {
        current_price = price;
    }
    else if (price < current_price) {
        current_price = price;
    }
    else {
        return ie_prices;
    }

    const auto delta_dir = current_price > previous_price ? 1 : -1;
    previous_price = current_price;

    if (current_price > ie_max_price) {
        ie_max_price = current_price;
        ie_delta_top = (ie_max_price - ie_start_price) / ie_start_price;
    }
    else if (current_price < ie_min_price) {
        ie_min_price = current_price;
        ie_delta_bot = (ie_start_price - ie_min_price) / ie_start_price;
    }
    
    const auto delta_down = (ie_max_price - current_price) / ie_max_price;
    const auto delta_up = (current_price - ie_min_price) / ie_min_price;

    if (ie_delta_top + delta_down >= BitBot::IntrinsicEvents::delta || ie_delta_bot + delta_up >= BitBot::IntrinsicEvents::delta) {
        auto remaining_delta = 0.0;
        auto ie_price = 0.0;

        if (delta_dir == 1) {
            remaining_delta = ie_delta_bot + delta_up;
            ie_price = ie_min_price * (1.0 + (BitBot::IntrinsicEvents::delta - ie_delta_bot));
        }
        else {
            remaining_delta = ie_delta_top + delta_down;
            ie_price = ie_max_price * (1.0 - (BitBot::IntrinsicEvents::delta - ie_delta_top));

        }

        while (remaining_delta >= 2 * BitBot::IntrinsicEvents::delta) {
            if (delta_dir == 1) {
                ie_max_price = std::min(ie_max_price, ie_price);
            }
            else {
                ie_min_price = std::max(ie_min_price, ie_price);
            }

            ie_prices.push_back(ie_price);

            const auto next_price = ie_price * (1.0 + delta_dir * BitBot::IntrinsicEvents::delta);
            ie_start_price = ie_price;

            if (delta_dir == 1) {
                ie_max_price = next_price;
                ie_min_price = ie_price;
            }
            else {
                ie_max_price = ie_price;
                ie_min_price = next_price;
            }

            ie_delta_top = (ie_max_price - ie_start_price) / ie_start_price;
            ie_delta_bot = (ie_start_price - ie_min_price) / ie_start_price;

            ie_price = next_price;
            remaining_delta -= BitBot::IntrinsicEvents::delta;
        }

        ie_prices.push_back(ie_price);

        ie_start_price = ie_price;
        ie_max_price = ie_price;
        ie_min_price = ie_price;
        ie_delta_top = 0.0;
        ie_delta_bot = 0.0;
    }

    return ie_prices;
}

void IntrinsicEvents::insert(BinanceKline binance_kline)
{
    for (const auto price : runner.step(binance_kline.open)) {
        events.push_back(IntrinsicEvent{ binance_kline.timestamp, (float)price });
    }
}

void IntrinsicEvents::insert(sptrBinanceKlines binance_klines)
{
    for (const auto& binance_kline : binance_klines->rows) {
        insert(binance_kline);
    }
}
