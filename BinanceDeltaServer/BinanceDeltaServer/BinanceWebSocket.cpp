
#include "DateTime.h"
#include "BinanceConstants.h"
#include "BinanceWebSocket.h"

#include "json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


BinanceWebSocket::BinanceWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
    websocket_thread_running(true),
    connected(false)
{
    ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
}

void BinanceWebSocket::start(void)
{
    // Start websocket worker
    websocket_thread = std::make_unique<std::thread>(&BinanceWebSocket::websocket_worker, this);
}

void BinanceWebSocket::shutdown(void)
{
    std::cout << "BinanceWebSocket: Shutting down" << std::endl;
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

void BinanceWebSocket::connect(void)
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
        auto const results = resolver.resolve(Binance::websocket::host, Binance::websocket::port);
        boost::asio::connect(websocket->next_layer().next_layer(), results.begin(), results.end());

        websocket->next_layer().handshake(boost::asio::ssl::stream_base::client);

        auto url = std::string{ Binance::websocket::url };

        for (auto symbol : Binance::symbols) {
            auto symbol_string = std::string{ symbol };
            std::transform(symbol_string.begin(), symbol_string.end(), symbol_string.begin(),
                [](unsigned char c) { return std::tolower(c); });

            url += symbol_string + "@trade/";
        }
        if (url.back() == '/') {
            url.pop_back();
        }

        websocket->handshake(Binance::websocket::host, url);

        connected = true;
    }
    catch (std::exception const& e) {
        connected = false;
    }
}

void BinanceWebSocket::websocket_worker(void)
{
    while (websocket_thread_running) {
        
        if (!connected) {
            connect();
        }

        if (!connected) {
            std::this_thread::sleep_for(500ms);
            continue;
        }

        //try {
            // Receive message
            auto buffer = boost::beast::flat_buffer{};
            websocket->read(buffer);
            const auto message_string = boost::beast::buffers_to_string(buffer.data());

            // Parse message
            auto error_message = std::string{};
            const auto message = json11::Json::parse(message_string, error_message);
            const auto action = message["action"].string_value();
            const auto table = message["table"].string_value();
            const auto data = message["data"];

            if (action == "insert" && table == "trade" && data.is_array()) {
                //std::cout << "BinanceWebSocket insert: " << message_string << std::endl;

                for (auto tick : data.array_items()) {
                    const auto symbol = tick["symbol"].string_value();
                    const auto price = tick["price"].number_value();
                    const auto volume = tick["size"].number_value();
                    const auto buy = tick["side"].string_value().compare("Buy") == 0;
                    const auto timestamp = DateTime::to_time_point(tick["timestamp"].string_value(), "%FT%TZ");

                    tick_data->append(symbol, timestamp, (float)price, (float)volume, buy);
                }
            }
            else {
                std::cout << "BinanceWebSocket rcv: " << message_string << std::endl;
            }
        //}
        //catch (std::exception const& e) {
        //    connected = false;
        //}
    }
}
