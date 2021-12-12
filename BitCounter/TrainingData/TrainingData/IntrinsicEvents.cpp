
#include "IntrinsicEvents.h"
#include "BitLib/Logger.h"
#include "Config.h"

#include <filesystem>


IntrinsicEvents::IntrinsicEvents(const Symbol& symbol)
{
    load(symbol);
}

std::istream& operator>>(std::istream& stream, IntrinsicEvent& row)
{
    stream.read(reinterpret_cast <char*> (&row.timestamp), sizeof(row.timestamp));
    stream.read(reinterpret_cast <char*> (&row.price), sizeof(row.price));
    stream.read(reinterpret_cast <char*> (&row.size), sizeof(row.size));
    stream.read(reinterpret_cast <char*> (&row.tick_id), sizeof(row.tick_id));
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

void IntrinsicEvents::load(const Symbol& symbol)
{
    events.clear();
    const auto file_path = std::string{ Config::base_path } + "\\intrinsic_events\\" + symbol.name.data() + ".dat";
    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        auto intrinsic_event = IntrinsicEvent{};
        data_file.read(reinterpret_cast <char*> (&delta), sizeof(delta));
        while (data_file >> intrinsic_event) {
            events.push_back(intrinsic_event);
        }
        data_file.close();
    }
}

void IntrinsicEvents::save_csv(std::string file_path)
{
    /*
    auto file = std::ofstream{};
    file.open(file_path);
    file << "price\n";
    for (auto row : events) {
        file << row.price << "\n";
    }
    file.close();
     */

    auto csv_file = std::ofstream{ file_path, std::ios::binary };
    csv_file << "\"timestamp\",\"price\",\"size\",\"tick_id\"\n";
    csv_file << std::fixed;
    for (const auto& ie : events) {
        const auto timestamp = ie.timestamp.time_since_epoch().count() / 1000000.0;
        csv_file.precision(6);
        csv_file << timestamp << ",";
        csv_file.precision(2);
        csv_file << ie.price << ",";
        csv_file.precision(3);
        csv_file << ie.price << ",";
        csv_file << ie.tick_id << "\n";
    }
    csv_file.close();
}

double IntrinsicEvents::get_delta(void)
{
    return delta;
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

    if (ie_delta_top + delta_down >= delta || ie_delta_bot + delta_up >= delta) {
        auto remaining_delta = 0.0;
        auto ie_price = 0.0;

        if (delta_dir == 1) {
            remaining_delta = ie_delta_bot + delta_up;
            ie_price = ie_min_price * (1.0 + (delta - ie_delta_bot));
        }
        else {
            remaining_delta = ie_delta_top + delta_down;
            ie_price = ie_max_price * (1.0 - (delta - ie_delta_top));
        }

        while (remaining_delta >= 2 * delta) {
            if (delta_dir == 1) {
                ie_max_price = std::min(ie_max_price, ie_price);
            }
            else {
                ie_min_price = std::max(ie_min_price, ie_price);
            }

            ie_prices.push_back(ie_price);

            const auto next_price = ie_price * (1.0 + delta_dir * delta);
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
            remaining_delta -= delta;
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

void calculate_thread(const Symbol& symbol, const TickData& tick_data)
{
    std::vector<IntrinsicEvent> events;
    events.clear();
    auto runner = IntrinsicEventRunner{ Config::IntrinsicEvents::delta };
    auto accum_size = 0.0f;
    auto tick_id = uint32_t{ 0 };
    for (const auto& [timestamp, price, size] : tick_data.rows) {
        accum_size += size;
        for (const auto price : runner.step(price)) {
            events.push_back(IntrinsicEvent{ timestamp, (float)price, accum_size, tick_id});
            accum_size = 0;
        }
        tick_id++;
    }

    auto file_path = std::string{ Config::base_path } + "\\intrinsic_events";
    std::filesystem::create_directories(file_path);
    file_path += std::string{ "\\" } + symbol.name.data() + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };

    data_file.write(reinterpret_cast<const char*>(&Config::IntrinsicEvents::delta), sizeof(Config::IntrinsicEvents::delta));
    for (auto&& row : events) {
        data_file.write(reinterpret_cast<const char*>(&row.timestamp), sizeof(row.timestamp));
        data_file.write(reinterpret_cast<const char*>(&row.price), sizeof(row.price));
        data_file.write(reinterpret_cast<const char*>(&row.size), sizeof(row.size));
        data_file.write(reinterpret_cast<const char*>(&row.tick_id), sizeof(row.tick_id));
        //data_file.write(reinterpret_cast<const char*>(&row.delta), sizeof(row.delta));
    }

    data_file.close();

    logger.info("Inserted %d events from %s, delta: %f", events.size(), symbol.name.data(), Config::IntrinsicEvents::delta);
}

void IntrinsicEvents::calculate_and_save(const Symbol& symbol, const TickData& tick_data)
{
    calculate_thread(symbol, tick_data);
    /*
    auto thread = std::thread{ calculate_thread, symbol, tick_data };
    threads.push_back(std::move(thread));

    if (threads.size() == (int)(std::thread::hardware_concurrency() * 0.8)) {
        threads.begin()->join();
        threads.erase(threads.begin());
    }
    */
}

void IntrinsicEvents::join(void)
{
    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();
}
