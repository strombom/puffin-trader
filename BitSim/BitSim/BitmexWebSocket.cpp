#include "pch.h"

#include "BitmexWebSocket.h"
#include "BitBotConstants.h"


BitmexWebSocket::BitmexWebSocket(void) :
    connected(false), websocket_thread_running(true)
{

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

        // Connect
        auto resolver = boost::asio::ip::tcp::resolver{ ioc };
        auto const results = resolver.resolve(BitSim::Trader::Bitmex::websocket_host, BitSim::Trader::Bitmex::websocket_port);
        boost::asio::connect(websocket->next_layer().next_layer(), results.begin(), results.end());
        websocket->next_layer().handshake(boost::asio::ssl::stream_base::client);
        websocket->handshake(BitSim::Trader::Bitmex::websocket_host, BitSim::Trader::Bitmex::websocket_url);

        // Subscribe to account status
        auto message = std::string{ "{\"op\": \"subscribe\", \"args\": [\"execution\",\"order\",\"margin\",\"position\",\"wallet\"]}" };
        std::cout << "subscribe: " << message << std::endl;
        websocket->write(boost::asio::buffer(message));

        // Receive Bitmex welcome message
        boost::beast::flat_buffer buffer;
        websocket->read(buffer);
        std::cout << boost::beast::make_printable(buffer.data()) << std::endl;

        connected = true;
    }
    catch (...) {
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
            std::this_thread::sleep_for(std::chrono::milliseconds{ 500 });
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

            std::cout << "BitmexWebSocket rcv: " << message_string << std::endl;

            /*
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
            */
        }
        catch (...) {
            connected = false;
        }
    }
}
