
#include "DateTime.h"
#include "BitmexConstants.h"
#include "BitmexWebSocket.h"

#include "json11.hpp"

#include <boost/beast/core/stream_traits.hpp>


BitmexWebSocket::BitmexWebSocket(sptrOrderBookData order_book_data) :
    order_book_data(order_book_data),
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
            
            if (action == "update" && table == "orderBook10" && data.is_array()) {
                for (auto data_symbol : data.array_items()) {
                    const auto symbol = data_symbol["symbol"].string_value();
                    const auto timestamp_ms = DateTime::to_time_point(data_symbol["timestamp"].string_value(), "%FT%TZ");
                    const auto top_bid = data_symbol["bids"].array_items()[0].array_items();
                    const auto top_ask = data_symbol["asks"].array_items()[0].array_items();

                    const auto bid_price = top_bid[0].number_value();
                    const auto bid_volume = top_bid[1].number_value();
                    const auto ask_price = top_ask[0].number_value();
                    const auto ask_volume = top_ask[1].number_value();

                    auto append = false;

                    if (previous_order_book.count(symbol) == 0) {
                        append = true;
                    }
                    else {
                        auto bid_volume_diff = std::abs(previous_order_book[symbol].bid_volume - bid_volume);
                        auto ask_volume_diff = std::abs(previous_order_book[symbol].ask_volume - ask_volume);

                        if (bid_volume == 0) {
                            bid_volume_diff = 1;
                        }
                        else {
                            bid_volume_diff /= bid_volume;
                        }

                        if (ask_volume == 0) {
                            ask_volume_diff = 1;
                        }
                        else {
                            ask_volume_diff /= ask_volume;
                        }

                        if (std::abs(previous_order_book[symbol].bid_price - bid_price) > 0.00001 ||
                            std::abs(previous_order_book[symbol].ask_price - ask_price) > 0.00001 ||
                            bid_volume_diff > 0.1 || ask_volume_diff > 0.1
                            ) {
                            append = true;
                        }
                    }

                    if (append) {
                        order_book_data->append(symbol, timestamp_ms, (float)bid_price, (float)bid_volume, (float)ask_price, (float)ask_volume);
                        previous_order_book[symbol].timestamp_ms = timestamp_ms.time_since_epoch().count();
                        previous_order_book[symbol].bid_price = (float)bid_price;
                        previous_order_book[symbol].bid_volume = (float)bid_volume;
                        previous_order_book[symbol].ask_price = (float)ask_price;
                        previous_order_book[symbol].ask_volume = (float)ask_volume;
                    }
                    else {
                        /*
                        std::cout << "nopend " <<
                            "sym(" << symbol << ") " <<
                            "ts(" << DateTime::to_string(timestamp_ms) << ") " <<
                            "bp(" << bid_price << ") " <<
                            "bv(" << bid_volume << ") " <<
                            "ap(" << ask_price << ") " <<
                            "av(" << ask_volume << ") " <<
                            std::endl;
                        */
                    }
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
