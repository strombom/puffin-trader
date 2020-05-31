
#include "DateTime.h"
#include "BitmexConstants.h"
#include "BitmexWebSocket.h"

#include <boost/beast/core/stream_traits.hpp>


BitmexWebSocket::BitmexWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
    //ctx(boost::asio::ssl::context::tlsv12_client),
    //websocket(boost::beast::websocket::stream<boost::asio::ip::tcp::socket>{ ioc, ctx }),
    websocket_thread_running(true)
{
    //auto config = web::websockets::client::websocket_client_config{};
    //client = std::make_unique<web::websockets::client::websocket_client>(config);
}

void BitmexWebSocket::start(void)
{
    boost::asio::io_context ioc;
    boost::asio::ssl::context ctx{ boost::asio::ssl::context::tlsv12_client };
    boost::asio::ip::tcp::resolver resolver{ ioc };
    //boost::beast::websocket::stream<boost::asio::ip::tcp::socket> websocket{ ioc, ctx };

    boost::beast::websocket::stream<boost::beast::ssl_stream<boost::asio::ip::tcp::socket>> websocket{ ioc, ctx };

    // Look up the domain name
    const auto port = "443";

    auto const results = resolver.resolve(host, port);
    boost::asio::connect(websocket.next_layer().next_layer(), results.begin(), results.end());

    websocket.next_layer().handshake(boost::asio::ssl::stream_base::client);

    // Set a decorator to change the User-Agent of the handshake
    websocket.set_option(boost::beast::websocket::stream_base::decorator(
        [](boost::beast::websocket::request_type& req)
        {
            req.set(boost::beast::http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                " websocket-client-coro");
        }));

    // Perform the websocket handshake
    websocket.handshake(host, url);

    // Subscribe to ticker symbols
    for (auto&& symbol : Bitmex::symbols) {
        //web::websockets::client::websocket_outgoing_message msg_out;
        auto message = std::string{ "{\"op\": \"subscribe\", \"args\": [\"trade:" } + symbol + std::string{ "\"]}" };
        std::cout << "subscribe: " << message << std::endl;
        //msg_out.set_utf8_message(message);
        websocket.write(boost::asio::buffer(message));
    }

    // This buffer will hold the incoming message
    boost::beast::flat_buffer buffer;

    // Read a message into our buffer
    websocket.read(buffer);

    // Close the WebSocket connection
    websocket.close(boost::beast::websocket::close_code::normal);

    // If we get here then the connection is closed gracefully

    // The make_printable() function helps print a ConstBufferSequence
    std::cout << boost::beast::make_printable(buffer.data()) << std::endl;

    // Start websocket worker
    //websocket_thread = std::make_unique<std::thread>(&BitmexWebSocket::websocket_worker, this);
}

void BitmexWebSocket::shutdown(void)
{
    std::cout << "BitmexWebSocket: Shutting down" << std::endl;
    websocket_thread_running = false;
    
    try {
        websocket_thread->join();
    }
    catch (...) {}
}

/*
bool BitmexWebSocket::json_test_field(const web::json::value& data, const std::string& name, const std::string& value)
{
    return data.has_field(U(name)) && data.at(U(name)).as_string().compare(U(value)) == 0;
}
*/

void BitmexWebSocket::websocket_worker(void)
{
    /*
    while (websocket_thread_running) {
        const auto response = client->receive().get();
        const auto body = response.extract_string().get();
        const auto data = web::json::value::parse(U(body));
        
        if (json_test_field(data, "action", "insert") && json_test_field(data, "table", "trade"))
        {
            const auto ticks = data.at(U("data")).as_array();

            for (auto tick : ticks) {
                const auto symbol = tick.at(U("symbol")).as_string();
                const auto price = tick.at(U("price")).as_double();
                const auto volume = tick.at(U("size")).as_double();
                const auto buy = tick.at(U("side")).as_string().compare(U("Buy")) == 0;
                const auto timestamp = DateTime::to_time_point_ms(tick.at(U("timestamp")).as_string(), "%FT%TZ");

                //std::wcout << "BitmexWebSocket: Insert table: " <<
                //    "timestamp(" << DateTime::to_string_iso_8601(timestamp) << ") " <<
                //    "symbol(" << symbol.c_str() << ") " <<
                //    "price(" << price << ") " <<
                //    "volume(" << volume << ") " <<
                //    "buy(" << buy << ") " << std::endl;

                tick_data->append(symbol, timestamp, (float) price, (float) volume, buy);
            }
        }
        else
        {
            std::wcout << "BitmexWebSocket: Rcv: " << data.to_string().c_str() << std::endl;
        }

    }
    */
}
