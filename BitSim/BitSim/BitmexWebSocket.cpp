#include "pch.h"

#include "Logger.h"
#include "DateTime.h"
#include "BitmexWebSocket.h"
#include "BitBotConstants.h"


BitmexWebSocket::BitmexWebSocket(void) :
    connected(false), websocket_thread_running(true)
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

    websocket->async_close(boost::beast::websocket::close_code::normal, boost::beast::bind_front_handler(&BitmexWebSocket::on_close, shared_from_this()));

    /*
    try {
        websocket->close(boost::beast::websocket::close_code::normal);
    }
    catch (...) {}
    */
}

void BitmexWebSocket::fail(boost::beast::error_code ec, const std::string &reason)
{
    logger.warn("BitmexWebSocket error: %s \"%s\"", reason, ec.message());
}

void BitmexWebSocket::connect(void)
{
    host_address = "";

    resolver = std::make_unique<boost::asio::ip::tcp::resolver>(boost::asio::make_strand(ioc));
    websocket = std::make_unique<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>>(boost::asio::make_strand(ioc), *ctx);

    resolver->async_resolve(
        BitSim::Trader::Bitmex::websocket_host,
        BitSim::Trader::Bitmex::websocket_port,
        boost::beast::bind_front_handler(&BitmexWebSocket::on_resolve, shared_from_this())
        //std::bind(&BitmexWebSocket::on_resolve, this)
        //boost::beast::bind_front_handler(&BitmexWebSocket::on_resolve, this); // shared_from_this())
    );
}

void BitmexWebSocket::on_resolve(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type results)
{
    if (ec) {
        fail(ec, "resolve");
        return;
    }

    boost::beast::get_lowest_layer(*websocket).expires_after(std::chrono::seconds(10));
    boost::beast::get_lowest_layer(*websocket).async_connect(results, boost::beast::bind_front_handler(&BitmexWebSocket::on_connect, shared_from_this()));
}

void BitmexWebSocket::on_connect(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type::endpoint_type ep)
{
    if (ec)
    {
        fail(ec, "connect");
        return;
    }        

    // Update the host_ string. This will provide the value of the host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    host_address = std::string{ BitSim::Trader::Bitmex::websocket_host } + ":" + std::to_string(ep.port());

    // Set a timeout on the operation
    boost::beast::get_lowest_layer(*websocket).expires_after(std::chrono::seconds(30));

    // Perform the SSL handshake
    websocket->next_layer().async_handshake(boost::asio::ssl::stream_base::client, boost::beast::bind_front_handler(&BitmexWebSocket::on_ssl_handshake, shared_from_this()));
}

void BitmexWebSocket::on_ssl_handshake(boost::beast::error_code ec)
{
    if (ec)
        return fail(ec, "ssl_handshake");

    // Turn off the timeout on the tcp_stream, because
    // the websocket stream has its own timeout system.
    boost::beast::get_lowest_layer(*websocket).expires_never();

    // Set suggested timeout settings for the websocket
    websocket->set_option(boost::beast::websocket::stream_base::timeout::suggested(boost::beast::role_type::client));

    // Set a decorator to change the User-Agent of the handshake
    websocket->set_option(boost::beast::websocket::stream_base::decorator(
        [](boost::beast::websocket::request_type& req)
        {
            req.set(boost::beast::http::field::user_agent,
                std::string(BOOST_BEAST_VERSION_STRING) +
                " websocket-client-async-ssl");
        }));

    // Perform the websocket handshake
    websocket->async_handshake(host_address, BitSim::Trader::Bitmex::websocket_url, boost::beast::bind_front_handler(&BitmexWebSocket::on_handshake, shared_from_this()));
}

void BitmexWebSocket::on_handshake(boost::beast::error_code ec)
{
    if (ec)
        return fail(ec, "handshake");

    // Send the message
    websocket->async_write(
        boost::asio::buffer("{\"message\"}"),
        boost::beast::bind_front_handler(
            &BitmexWebSocket::on_write,
            shared_from_this()));
}

void BitmexWebSocket::on_write(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    if (ec)
        return fail(ec, "write");

    // Read a message into our buffer
    websocket->async_read(websocket_buffer, boost::beast::bind_front_handler(&BitmexWebSocket::on_read, shared_from_this()));
}

void BitmexWebSocket::on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    if (ec)
        return fail(ec, "read");

    // Close the WebSocket connection
    websocket->async_close(boost::beast::websocket::close_code::normal, boost::beast::bind_front_handler(&BitmexWebSocket::on_close, shared_from_this()));
}

void BitmexWebSocket::on_close(boost::beast::error_code ec)
{
    if (ec)
        return fail(ec, "close");

    // If we get here then the connection is closed gracefully

    // The make_printable() function helps print a ConstBufferSequence
    std::cout << "on_close: " << boost::beast::make_printable(websocket_buffer.data()) << std::endl;
}

void BitmexWebSocket::websocket_worker(void)
{
    while (websocket_thread_running) {
        logger.info("BitmexWebSocket:websocket_worker: connect");
        connect();
        logger.info("BitmexWebSocket:websocket_worker: ioc.run start");
        ioc.run();
        logger.info("BitmexWebSocket:websocket_worker: ioc.run end");

        websocket_thread_running = false;
    }
}


/*
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
catch (...) {
    connected = false;
}
    */

    
    /*
    try {
        resolver = std::make_unique<boost::asio::ip::tcp::resolver>(boost::asio::make_strand(ioc));
        websocket = std::make_unique<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>>(boost::asio::make_strand(ioc), *ctx);

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


        // Receive Bitmex welcome message
        boost::beast::flat_buffer buffer;
        websocket->read(buffer);
        std::cout << boost::beast::make_printable(buffer.data()) << std::endl;

        // Authenticate
        const auto auth_expires = std::to_string(std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now()).time_since_epoch().count());

        auto aa = json11::Json::array{"abc"};
        //std::string{BitSim::Trader::Bitmex::api_key_id}

        json11::Json auth_command = json11::Json::object{
            { "op", "authKeyExpires" },
            { "args", json11::Json::array{aa, auth_expires} }
        };

        std::cout << auth_command.dump() << std::endl;

        //auto message = std::string{ "{\"op\": \"authKeyExpires\", \"args\": [\"execution\",\"order\",\"margin\",\"position\",\"wallet\"]}" };
        //std::cout << "subscribe: " << message << std::endl;
        //websocket->write(boost::asio::buffer(message));

        //websocket->read(buffer);
        //std::cout << boost::beast::make_printable(buffer.data()) << std::endl;
        
        // Subscribe to account status
        //auto message = std::string{ "{\"op\": \"subscribe\", \"args\": [\"execution\",\"order\",\"margin\",\"position\",\"wallet\"]}" };
        //std::cout << "subscribe: " << message << std::endl;
        //websocket->write(boost::asio::buffer(message));

        connected = true;
    }
    catch (...) {
        connected = false;
    }
    */

