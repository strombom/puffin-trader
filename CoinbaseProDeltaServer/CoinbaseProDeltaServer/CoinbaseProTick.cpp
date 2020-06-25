
#include "CoinbaseProConstants.h"
#include "CoinbaseProTick.h"

#include "BitLib/DateTime.h"

#include <iostream>


using namespace std::chrono_literals;

CoinbaseProTick::CoinbaseProTick(sptrTickData tick_data) :
    tick_data(tick_data),
    tick_thread_running(true)
{
    rest_api = std::make_unique<CoinbaseProRestApi>();
}

void CoinbaseProTick::start(void)
{
    for (const auto symbol : CoinbasePro::symbols) {
        const auto last_id = rest_api->get_aggregate_trades(tick_data, symbol, -1);
        last_ids[symbol] = last_id;
    }

    // Start tick worker
    tick_thread = std::make_unique<std::thread>(&CoinbaseProTick::tick_worker, this);
}

void CoinbaseProTick::shutdown(void)
{
    std::cout << "CoinbaseProTick: Shutting down" << std::endl;
    tick_thread_running = false;

    try {
        tick_thread->join();
    }
    catch (...) {}
}

void CoinbaseProTick::tick_worker(void)
{
    while (tick_thread_running) {

        std::this_thread::sleep_for(300ms);

            for (const auto symbol : CoinbasePro::symbols) {
                try {
                    last_ids[symbol] = rest_api->get_aggregate_trades(tick_data, symbol, last_ids[symbol]);
                }
                catch (std::exception const& e) {

                }
            }



        /*
        try {
            // Receive message
            auto buffer = boost::beast::flat_buffer{};
            websocket->read(buffer);
            const auto message_string = boost::beast::buffers_to_string(buffer.data());

            // Parse message
            auto error_message = std::string{};
            const auto message = json11::Json::parse(message_string, error_message);
            if (!message["stream"].is_string() || !message["data"].is_object()) {
                continue;
            }
            const auto stream = message["stream"].string_value();
            auto find_stream_type = stream.find("@aggTrade");
            if (find_stream_type == std::string::npos || find_stream_type == 0) {
                continue;
            }
            auto symbol = stream.substr(0, find_stream_type);
            std::transform(symbol.begin(), symbol.end(), symbol.begin(), [](unsigned char c) { return std::toupper(c); });

            const auto timestamp_raw_ms = (long long)message["data"]["T"].number_value();
            const auto price_raw = message["data"]["p"].string_value();
            const auto volume_raw = message["data"]["q"].string_value();

            const auto timestamp = time_point_ms{ std::chrono::milliseconds{timestamp_raw_ms} };
            const auto price = std::stod(price_raw);
            const auto volume = std::stod(volume_raw);
            const auto buy = message["data"]["m"].bool_value();

            if (timestamp_raw_ms == 0 || price == 0.0 || volume == 0.0 || !message["data"]["m"].is_bool()) {
                continue;
            }

            tick_data->append(symbol, timestamp, (float)price, (float)volume, buy);
        }
        catch (std::exception const& e) {
            connected = false;
        }
        */
    }
}
