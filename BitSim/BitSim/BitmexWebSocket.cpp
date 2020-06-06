#include "pch.h"

#include "Logger.h"
#include "DateTime.h"
#include "BitmexWebSocket.h"
#include "BitBotConstants.h"


BitmexWebSocket::BitmexWebSocket(sptrBitmexAccount bitmex_account) :
    connected(false), websocket_thread_running(true),
    bitmex_account(bitmex_account)
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

    // Close the WebSocket connection
    websocket->async_close(boost::beast::websocket::close_code::normal, boost::beast::bind_front_handler(&BitmexWebSocket::on_close, shared_from_this()));

    try {
        websocket_thread->join();
    }
    catch (...) {}

    websocket->async_close(boost::beast::websocket::close_code::normal, boost::beast::bind_front_handler(&BitmexWebSocket::on_close, shared_from_this()));
}

void BitmexWebSocket::fail(boost::beast::error_code ec, const std::string &reason)
{
    logger.warn("BitmexWebSocket error: %s \"%s\"", reason.c_str(), ec.message().c_str());
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
    );
}

void BitmexWebSocket::send(const std::string &message)
{
    logger.info("BitmexWebSocket::send: %s", message.c_str());

    websocket->async_write(boost::asio::buffer(message), boost::beast::bind_front_handler(&BitmexWebSocket::on_write, shared_from_this()));
}

void BitmexWebSocket::request_authentication(void)
{
    const auto expires = authenticator.generate_expiration(BitSim::Trader::Bitmex::auth_timeout);
    const auto sign_message = std::string{ "GET" } + BitSim::Trader::Bitmex::websocket_url + std::to_string(expires);
    const auto signature = authenticator.authenticate(sign_message);

    json11::Json auth_command = json11::Json::object{
        { "op", "authKeyExpires" },
        { "args", json11::Json::array{BitSim::Trader::Bitmex::api_key, (int)expires, signature} }
    };

    send(auth_command.dump());
}

void BitmexWebSocket::parse_message(const std::string& message)
{
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto command = json11::Json::parse(message.c_str(), error_message);

    if (command["info"].string_value() == "Welcome to the BitMEX Realtime API.") {
        request_authentication();
    }
    else if (command["request"]["op"].string_value() == "authKeyExpires") {
        if (command["success"].bool_value() == true) {
            json11::Json subscribe_command = json11::Json::object{
                { "op", "subscribe" },
                { "args", json11::Json::array{"order", "position", "wallet", "trade:XBTUSD"} }
                // "margin", "execution", "transact", 
            };
            send(subscribe_command.dump());
        }
        else {
            std::this_thread::sleep_for(2s);
            request_authentication();
        }
    }
    else if (command["table"].string_value() == "order") {
        const auto& action = command["action"].string_value();

        for (const auto& data : command["data"].array_items()) {

            const auto& order_id = data["orderID"].string_value();
            const auto& symbol = data["symbol"].string_value();
            const auto timestamp = DateTime::to_time_point_ms(data["timestamp"].string_value());

            if (action == "insert") {
                const auto& buy = (data["side"].string_value() == "Buy");
                const auto order_size = data["orderQty"].int_value();
                const auto price = data["price"].number_value();
                bitmex_account->insert_order(symbol, order_id, timestamp, buy, order_size, price);
            }
            else if (action == "update" && data["ordStatus"].string_value() == "Filled") {
                const auto& remaining_size = data["leavesQty"].int_value();
                bitmex_account->fill_order(symbol, order_id, timestamp, remaining_size);
            }
            else  if (action == "delete") {
                bitmex_account->delete_order(order_id);
            }
        }
    }
    else if (command["table"].string_value() == "trade") {
        for (const auto& data : command["data"].array_items()) {
            const auto& symbol = data["symbol"].string_value();
            const auto price = data["price"].number_value();
            bitmex_account->set_price(symbol, price);
        }
    }
    else if (command["table"].string_value() == "position") {
        //logger.info("BitmexWebSocket::parse_message: position (%s)", message.c_str());
        for (const auto& data : command["data"].array_items()) {
            if (data["markValue"].is_number()) {
                const auto mark_value = data["markValue"].number_value();
                bitmex_account->set_leverage(mark_value);
            }
        }
    }
    else if (command["table"].string_value() == "wallet") {
        //logger.info("BitmexWebSocket::parse_message: wallet (%s)", message.c_str());
        for (const auto& data : command["data"].array_items()) {
            const auto amount = data["amount"].number_value() / 100000000; // Convert from Satoshis to Bitcoin
            bitmex_account->set_wallet(amount);
        }
    }
    else if (command["table"].string_value() == "margin") {

    }
    else if (command["table"].string_value() == "execution") {

    }
    else if (command["table"].string_value() == "transact") {

    }
    else if (command["subscribe"].is_string() && command["success"].bool_value()) {

    }
    else {
        logger.info("BitmexWebSocket::parse_message: unknown command (%s)", message.c_str());
    }
}

void BitmexWebSocket::on_resolve(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type results)
{
    logger.info("BitmexWebSocket::on_resolve");

    if (ec) {
        fail(ec, "on_resolve");
        return;
    }

    boost::beast::get_lowest_layer(*websocket).expires_after(std::chrono::seconds(10));
    boost::beast::get_lowest_layer(*websocket).async_connect(results, boost::beast::bind_front_handler(&BitmexWebSocket::on_connect, shared_from_this()));
}

void BitmexWebSocket::on_connect(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type::endpoint_type ep)
{
    logger.info("BitmexWebSocket::on_connect");

    if (ec)
    {
        fail(ec, "on_connect");
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
    logger.info("BitmexWebSocket::on_ssl_handshake");

    if (ec) {
        fail(ec, "on_ssl_handshake");
        return;
    }

    // Turn off the timeout on the tcp_stream, because
    // the websocket stream has its own timeout system.
    boost::beast::get_lowest_layer(*websocket).expires_never();

    // Enable keep-alive pings
    websocket->set_option(boost::beast::websocket::stream_base::timeout{
            std::chrono::seconds(30),
            std::chrono::seconds(10),
            true
        });

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
    logger.info("BitmexWebSocket::on_handshake");

    if (ec) {
        fail(ec, "on_handshake");
        return;
    }

    websocket_buffer.clear();
    websocket->async_read(websocket_buffer, boost::beast::bind_front_handler(&BitmexWebSocket::on_read, shared_from_this()));
}

void BitmexWebSocket::on_write(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    logger.info("BitmexWebSocket::on_write");

    if (ec) {
        return fail(ec, "write");
    }
}

void BitmexWebSocket::on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    if (ec) {
        if (!websocket_thread_running) {
            // Application shutting down
            return;
        }
        else {
            return fail(ec, "read");
        }
    }

    auto ss = std::stringstream{};
    ss << boost::beast::make_printable(websocket_buffer.data());
    const auto message = ss.str();

    //logger.info("BitmexWebSocket::on_read (%d): %s", (int) bytes_transferred, message.c_str());


    websocket_buffer.clear();
    websocket->async_read(websocket_buffer, boost::beast::bind_front_handler(&BitmexWebSocket::on_read, shared_from_this()));

    parse_message(message);
}

void BitmexWebSocket::on_close(boost::beast::error_code ec)
{
    if (ec) {
        return fail(ec, "close");
    }
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
