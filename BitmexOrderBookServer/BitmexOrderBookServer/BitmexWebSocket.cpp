
#include "DateTime.h"
#include "BitmexConstants.h"
#include "BitmexWebSocket.h"

#include "json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


BitmexWebSocket::BitmexWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
    websocket_thread_running(true),
    connected(false)
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
            auto message = std::string{ "{\"op\": \"subscribe\", \"args\": [\"orderBook10:" } + symbol + std::string{ "\"]}" };
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
            
/*
subscribe: {"op": "subscribe", "args": ["orderBook10:XBTUSD"]}
{"info":"Welcome to the BitMEX Realtime API.","version":"2020-08-28T23:43:16.000Z","timestamp":"2020-09-01T19:06:42.456Z","docs":"https://www.bitmex.com/app/wsAPI","limit":{"remaining":39}}
BitmexWebSocket rcv: {"success":true,"subscribe":"orderBook10:XBTUSD","request":{"op":"subscribe","args":["orderBook10:XBTUSD"]}}
BitmexWebSocket rcv: {"table":"orderBook10","action":"partial","keys":["symbol"],"types":{"symbol":"symbol","bids":"","asks":"","timestamp":"timestamp"},"foreignKeys":{"symbol":"instrument"},"attributes":{"symbol":"sorted"},"filter":{"symbol":"XBTUSD"},"data":[{"symbol":"XBTUSD","bids":[[12005,2875434],[12004.5,65816],[12004,267135],[12003.5,40963],[12003,351],[12002.5,22126],[12002,7496],[12001.5,124930],[12001,28151],[12000.5,9631]],"asks":[[12005.5,822789],[12006,2788],[12006.5,9321],[12007,9964],[12007.5,35349],[12008,32062],[12008.5,117476],[12009,63946],[12009.5,74958],[12010,44228]],"timestamp":"2020-09-01T19:06:42.413Z"}]}
BitmexWebSocket rcv: {"table":"orderBook10","action":"update","data":[{"symbol":"XBTUSD","bids":[[12005,2875434],[12004.5,65816],[12004,267135],[12003.5,40963],[12003,351],[12002.5,22126],[12002,7662],[12001.5,124930],[12001,28151],[12000.5,9631]],"timestamp":"2020-09-01T19:06:42.599Z","asks":[[12005.5,822789],[12006,2788],[12006.5,9321],[12007,9964],[12007.5,35349],[12008,32062],[12008.5,117476],[12009,63946],[12009.5,74958],[12010,44228]]}]}
*/

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
