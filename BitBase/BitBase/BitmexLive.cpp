#include "pch.h"

#include "BitBotConstants.h"
#include "BitmexLive.h"
#include "DateTime.h"
#include "Logger.h"

#include <array>
#include <regex>
#include <string>
#include <iostream>


BitmexLive::BitmexLive(sptrDatabase database, tick_data_updated_callback_t tick_data_updated_callback) :
    database(database), tick_data_updated_callback(tick_data_updated_callback),
    state(BitmexLiveState::idle), tick_data_thread_running(true)
{
    zmq_client = std::make_unique<zmq::socket_t>(zmq_context, zmq::socket_type::req);
    zmq_client->connect(BitBase::Bitmex::Live::address);

    tick_data_worker_thread = std::make_unique<std::thread>(&BitmexLive::tick_data_worker, this);
}

BitmexLiveState BitmexLive::get_state(void)
{
    return state;
}

void BitmexLive::shutdown(void)
{
    logger.info("BitmexLive::shutdown");

    {
        // Will not start new downloads after this section
        //auto slock = std::scoped_lock{ start_download_mutex };
        //state = BitmexDailyState::idle;
    }

    tick_data_thread_running = false;
    tick_data_condition.notify_all();

    //download_manager->abort_client(BitBase::Bitmex::Daily::downloader_client_id);

    try {
        tick_data_worker_thread->join();
    }
    catch (...) {}
}

void BitmexLive::start(void)
{
    assert(state == BitmexLiveState::idle);
    
    state = BitmexLiveState::downloading;

    tick_data_condition.notify_one();

    //for (int i = 0; i < BitBase::Bitmex::Daily::active_downloads_max; ++i) {
    //    start_next_download();
    //}
}

void BitmexLive::tick_data_worker(void)
{
    while (tick_data_thread_running) {
        {
            auto tick_data_lock = std::unique_lock<std::mutex>{ tick_data_mutex };
            tick_data_condition.wait(tick_data_lock);
        }


        while (tick_data_thread_running) {

            for (auto&& symbol : BitBase::Bitmex::symbols) {

                const auto timestamp_next = database->get_attribute(BitBase::Bitmex::exchange_name, symbol, "tick_data_last_timestamp", BitBase::Bitmex::first_timestamp);

                json11::Json command = json11::Json::object{
                    { "command", "get_ticks" },
                    { "symbol", symbol },
                    { "timestamp_start", DateTime::to_string_iso_8601(timestamp_next) }
                };

                auto message = zmq::message_t{ command.dump() };
                auto result = zmq_client->send(message, zmq::send_flags::dontwait);

                if (result.has_value()) {
                    zmq_client->recv(message);
                    auto intervals_buffer = std::stringstream{ std::string(static_cast<char*>(message.data()), message.size()) };
                    std::cout << "Recv: " << std::string(static_cast<char*>(message.data()), message.size()) << std::endl;
                    //auto intervals = Intervals{ timestamp_start , interval };
                    //intervals_buffer >> intervals;
                }
            }

            break;


            /*


            auto tick_data = uptrTickData{};
            {
                auto slock = std::scoped_lock{ tick_data_mutex };
                if (!tick_data_queue.empty()) {
                    tick_data = std::move(tick_data_queue.front());
                    tick_data_queue.pop_front();
                }
                else {
                    break;
                }
            }

            auto timer = Timer{};
            auto symbol_names = std::unordered_set<std::string>{};
            for (auto&& symbol_tick_data = tick_data->begin(); symbol_tick_data != tick_data->end(); ++symbol_tick_data) {
                const auto symbol_name = symbol_tick_data->first;
                symbol_names.insert(symbol_name);
                database->extend_tick_data(BitBase::Bitmex::exchange_name, symbol_name, std::move(symbol_tick_data->second), BitBase::Bitmex::first_timestamp);
            }
            //update_symbol_names(symbol_names);
            tick_data_updated_callback();

            logger.info("BitmexDaily::tick_data_worker tick_data appended to database (%d ms)", timer.elapsed().count()/1000);
            */
        }
    }
    logger.info("BitmexDaily::tick_data_worker exit");
}

BitmexLive::uptrTickData BitmexLive::parse_raw(const std::stringstream& raw_data)
{
    auto timer = Timer{};

    auto tick_data = std::make_unique<TickData>();

    const auto linesregx = std::regex{ "\\n" };
    const auto indata = std::string{ raw_data.str() };
    auto row_it = std::sregex_token_iterator{ indata.begin(), indata.end(), linesregx, -1 };
    auto row_end = std::sregex_token_iterator{};

    ++row_it; // Skip table headers
    while (row_it != row_end) {
        const auto row = std::string{ row_it->str() };
        ++row_it;

        if (row.length() < 40) {
            return nullptr;
        }

        auto timestamp = time_point_ms{};
        auto ss = std::istringstream{ row.substr(0, 29) };
        ss >> date::parse("%FD%T", timestamp);
        if (ss.fail()) {
            return nullptr;
        }

        auto commas = std::array<int, 5>{ 0, 0, 0, 0, 0 };
        auto cidx = 0;
        auto p = 29;

        while (cidx < 5 && p < row.length()) {
            const auto c = row.at(p);
            if (c == '\n') {
                break;
            }
            else if (c == ',') {
                commas[cidx] = p + 1;
                ++cidx;
            }
            ++p;
        }

        if (commas[0] != 30 || cidx != 5) {
            return nullptr;
        }

        const std::string symbol = row.substr(commas[0], (size_t)(commas[1] - commas[0] - 1));

        auto buy = false;
        if (row.substr(commas[1], (size_t) (commas[2] - commas[1] - 1)) == "Buy") {
            buy = true;
        }

        auto volume = float{};
        try {
            const std::string token = row.substr(commas[2], (size_t)(commas[3] - commas[2] - 1));
            volume = std::stof(token);
        }
        catch (...) {
            return nullptr; // Invalid volume format
        }

        auto price = float{};
        try {
            const std::string token = row.substr(commas[3], (size_t)(commas[4] - commas[3] - 1));
            price = std::stof(token);
        }
        catch (...) {
            return nullptr; // Invalid price format
        }

        if ((*tick_data)[symbol] == nullptr) {
            (*tick_data)[symbol] = std::make_unique<Ticks>();
        }
        (*tick_data)[symbol]->rows.push_back({ timestamp, price, volume, buy });
    }

    logger.info("BitmexDaily::parse_raw end (%d ms)", timer.elapsed().count() / 1000);

    return tick_data;
}
