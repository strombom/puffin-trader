
#include "DateTime.h"
#include "BitmexConstants.h"
#include "BitmexWebSocket.h"

#include "json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


BitmexWebSocket::BitmexWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
    connected(false),
    websocket_thread_running(true)
{
    ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
}

void BitmexWebSocket::start(void)
{
    // Start websocket worker
    websocket_thread = std::make_unique<std::thread>(&BitmexWebSocket::websocket_worker, this);
}

void BitmexWebSocket::shutdown(void)
{
    std::cout << "BitmexWebSocket: Shutting down" << std::endl;
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

void BitmexWebSocket::connect(void)
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
        auto const results = resolver.resolve(host, port);
        boost::asio::connect(websocket->next_layer().next_layer(), results.begin(), results.end());

        websocket->next_layer().handshake(boost::asio::ssl::stream_base::client);

        websocket->handshake(host, url);

        // Subscribe to ticker symbols
        for (auto&& symbol : Bitmex::symbols) {
            //web::websockets::client::websocket_outgoing_message msg_out;
            auto message = std::string{ "{\"op\": \"subscribe\", \"args\": [\"trade:" } + symbol + std::string{ "\"]}" };
            std::cout << "subscribe: " << message << std::endl;
            //msg_out.set_utf8_message(message);
            websocket->write(boost::asio::buffer(message));
        }

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

void BitmexWebSocket::websocket_worker(void)
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
            const auto action = message["action"].string_value();
            const auto table = message["table"].string_value();
            const auto data = message["data"];

            if (action == "insert" && table == "trade" && data.is_array()) {
                //std::cout << "BitmexWebSocket insert: " << message_string << std::endl;

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
                std::cout << "BitmexWebSocket rcv: " << message_string << std::endl;
            }
        }
        catch (std::exception const& e) {
            connected = false;
        }
    }
}
