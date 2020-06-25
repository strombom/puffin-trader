
#include "CoinbaseConstants.h"
#include "CoinbaseTick.h"

#include "BitLib/DateTime.h"
#include "BitLib/json11/json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


using namespace std::chrono_literals;

CoinbaseTick::CoinbaseTick(sptrTickData tick_data) :
    tick_data(tick_data),
    websocket_thread_running(true),
    connected(false)
{
    ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
}

void CoinbaseTick::start(void)
{
    // Start websocket worker
    websocket_thread = std::make_unique<std::thread>(&CoinbaseTick::websocket_worker, this);
}

void CoinbaseTick::shutdown(void)
{
    std::cout << "CoinbaseTick: Shutting down" << std::endl;
    websocket_thread_running = false;

    try {
        websocket_thread->join();
    }
    catch (...) {}

    try {
        websocket->close(boost::beast::websocket::close_code::normal);
    }
    catch (...) {}
}

void CoinbaseTick::connect(void)
{
    try {
        websocket = std::make_unique<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>>>(ioc, *ctx);

        websocket->set_option(boost::beast::websocket::stream_base::decorator(
            [](boost::beast::websocket::request_type& req)
            {
                req.set(boost::beast::http::field::user_agent,
                    std::string(BOOST_BEAST_VERSION_STRING) +
                    " websocket-client-coro");
            }));

        websocket->set_option(boost::beast::websocket::stream_base::timeout{
                std::chrono::seconds(20),
                std::chrono::seconds(10),
                true
            });

        auto resolver = boost::asio::ip::tcp::resolver{ ioc };
        auto const results = resolver.resolve(Coinbase::websocket::host, Coinbase::websocket::port);
        boost::asio::connect(websocket->next_layer().next_layer(), results.begin(), results.end());

        websocket->next_layer().handshake(boost::asio::ssl::stream_base::client);

        auto url = std::string{ Coinbase::websocket::url };

        for (auto symbol : Coinbase::symbols) {
            auto symbol_string = std::string{ symbol };
            std::transform(symbol_string.begin(), symbol_string.end(), symbol_string.begin(), [](unsigned char c) { return std::tolower(c); });
            url += symbol_string + "@aggTrade/";
        }
        if (url.back() == '/') {
            url.pop_back();
        }

        websocket->handshake(Coinbase::websocket::host, url);

        connected = true;
    }
    catch (std::exception const& e) {
        connected = false;
    }
}

void CoinbaseTick::websocket_worker(void)
{
    while (websocket_thread_running) {

        if (!connected) {
            connect();
        }

        if (!connected) {
            std::this_thread::sleep_for(500ms);
            continue;
        }

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
    }
}
