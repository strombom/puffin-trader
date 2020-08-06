
#include "CoinbaseProConstants.h"
#include "CoinbaseProWebSocket.h"
#include "BitLib/DateTime.h"
#include "BitLib/json11/json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


CoinbaseProWebSocket::CoinbaseProWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
    websocket_thread_running(true),
    connected(false)
{
    ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
}

void CoinbaseProWebSocket::start(void)
{
    // Start websocket worker
    websocket_thread = std::make_unique<std::thread>(&CoinbaseProWebSocket::websocket_worker, this);
}

void CoinbaseProWebSocket::shutdown(void)
{
    std::cout << "CoinbaseProWebSocket: Shutting down" << std::endl;
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

void CoinbaseProWebSocket::connect(void)
{
    try
    {
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
        auto const results = resolver.resolve(CoinbasePro::WebSocket::host, CoinbasePro::WebSocket::port);
        boost::asio::connect(websocket->next_layer().next_layer(), results.begin(), results.end());

        // Set SNI Hostname (many hosts need this to handshake successfully)
        if (!SSL_set_tlsext_host_name(websocket->next_layer().native_handle(), CoinbasePro::WebSocket::host))
        {
            boost::system::error_code ec{ static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category() };
            throw boost::system::system_error{ ec };
        }

        websocket->next_layer().handshake(boost::asio::ssl::stream_base::client);
        websocket->handshake(CoinbasePro::WebSocket::host, std::string{ CoinbasePro::WebSocket::url });

        // Subscribe to ticker symbols
        auto message = std::string{ "{\"type\":\"subscribe\"," };
        message += "\"product_ids\":[";
        for (auto&& symbol : CoinbasePro::symbols) {
            message += "\"" + std::string{ symbol } + "\",";
        }
        if (message.back() == ',') {
            // Remove last comma
            message.pop_back();
        }
        message += "],";
        message += "\"channels\":[";
        message += "{\"name\":\"ticker\",";
        message += "\"product_ids\":[";
        for (auto&& symbol : CoinbasePro::symbols) {
            message += "\"" + std::string{ symbol } + "\",";
        }
        if (message.back() == ',') {
            // Remove last comma
            message.pop_back();
        }
        message += "]}]}";
        websocket->write(boost::asio::buffer(message));

        // Receive Bitmex welcome message
        boost::beast::flat_buffer buffer;
        websocket->read(buffer);
        std::cout << boost::beast::make_printable(buffer.data()) << std::endl;

        connected = true;
    }
    catch (std::exception const& e) {
        connected = false;
    }
}

void CoinbaseProWebSocket::websocket_worker(void)
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

            if (!message["type"].is_string() || message["type"].string_value().compare("ticker") != 0) {
                continue;
            }
            
            const auto symbol = message["product_id"].string_value();
            const auto timestamp = DateTime::iso8601_us_to_time_point_ms(message["time"].string_value());
            const auto price = std::stod(message["price"].string_value());
            const auto volume = std::stod(message["last_size"].string_value());
            const auto buy = message["side"].string_value().compare("buy") == 0;
            const auto trade_id = (long long)message["trade_id"].number_value();

            tick_data->append(symbol, timestamp, (float)price, (float)volume, buy, trade_id);
        }
        catch (std::exception const& e) {
            connected = false;
        }
    }
}
