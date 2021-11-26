#include "ByBitConfig.h"
#include "ByBitWebsocket.h"
#include "BitLib/Logger.h"
#include "BitLib/DateTime.h"
#include "Symbols.h"


ByBitWebSocket::ByBitWebSocket(const std::string& url, bool authenticate, std::vector<std::string> topics, sptrOrderManager order_manager) :
    url(url), authenticate(authenticate), topics(topics), order_manager(order_manager), connected(false), websocket_thread_running(true), heartbeat_thread_running(true)
{
    ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12_client);
}

void ByBitWebSocket::start(void)
{
    websocket_thread = std::make_unique<std::thread>(&ByBitWebSocket::websocket_worker, this);
    heartbeat_thread = std::make_unique<std::thread>(&ByBitWebSocket::heartbeat_worker, this);
}

void ByBitWebSocket::shutdown(void)
{
    logger.warn("ByBitWebSocket Shutting down");

    // Close the WebSocket connection
    websocket_thread_running = false;
    websocket->async_close(boost::beast::websocket::close_code::normal, boost::beast::bind_front_handler(&ByBitWebSocket::on_close, shared_from_this()));

    try {
        heartbeat_thread->join();
        websocket_thread->join();
    }
    catch (...) {}
}

void ByBitWebSocket::send_heartbeat(void)
{
    auto auth_command = (json11::Json) json11::Json::object{ { "op", "ping" } };
    send(auth_command.dump());
}

void ByBitWebSocket::fail(boost::beast::error_code ec, const std::string& reason)
{
    logger.warn("ByBitWebSocket error: %s \"%s\"", reason.c_str(), ec.message().c_str());
}

void ByBitWebSocket::connect(void)
{
    static bool first = true;
    if (first) {
        first = false;
    }
    else {
        // Rate limit reconnects
        std::this_thread::sleep_for(ByBit::WebSocket::reconnect_delay);
    }

    ioc.reset();
    resolver = std::make_unique<boost::asio::ip::tcp::resolver>(boost::asio::make_strand(ioc));
    websocket = std::make_unique<boost::beast::websocket::stream<boost::beast::ssl_stream<boost::beast::tcp_stream>>>(boost::asio::make_strand(ioc), *ctx);

    resolver->async_resolve(
        ByBit::WebSocket::host,
        ByBit::WebSocket::port,
        boost::beast::bind_front_handler(&ByBitWebSocket::on_resolve, shared_from_this())
    );
}

void ByBitWebSocket::send(const std::string& message)
{
    auto lock = std::scoped_lock{ send_mutex };
    //logger.info("ByBitWebSocket::send: %s", message.c_str());

    websocket->async_write(boost::asio::buffer(message), boost::beast::bind_front_handler(&ByBitWebSocket::on_write, shared_from_this()));
}

void ByBitWebSocket::request_authentication(void)
{
    const auto expires = std::to_string(authenticator.generate_expiration(ByBit::WebSocket::auth_timeout));
    const auto sign_message = std::string{ "GET/realtime" } + expires;
    const auto signature = authenticator.authenticate(sign_message);

    json11::Json auth_command = json11::Json::object{
        { "op", "auth" },
        { "args", json11::Json::array{ByBit::api_key, expires, signature} }
    };

    send(auth_command.dump());
}

void ByBitWebSocket::subscribe(void)
{
    json11::Json subscribe_command = json11::Json::object{
        { "op", "subscribe" },
        { "args", topics }
    };
    send(subscribe_command.dump());

}

void ByBitWebSocket::parse_message(std::string *message)
{
    message->reserve(message->size() + simdjson::SIMDJSON_PADDING);
    auto doc = json_parser.iterate(*message);

    if (message->compare(0, 6, "{\"succ") == 0) {
        const bool success = doc["success"];
        const std::string_view ret_msg = doc["ret_msg"];
        const std::string_view op = doc["request"]["op"];

        if (authenticate && op == "auth") {
            if (success) {
                logger.info("Authenticated");
                subscribe();
            }
            else {
                request_authentication();
            }
        }
        else if (op == "subscribe") {
            // Todo: Reset resubscribe timeout?
        }
        else if (op == "ping") {
            if (ret_msg == "pong") {

            }
        }
        else {
            auto b = 1;
        }
        return;
    }

    const std::string_view topic = doc["topic"];

    if (topic == "order") {
        for (auto order : doc["data"]) {
            const std::string_view id_str = order["order_link_id"];
            if (id_str.size() == 36) {
                const auto id = Uuid{ id_str };
                const auto symbol = string_to_symbol(order["symbol"]);
                const auto side = string_to_side(order["side"]);
                const auto price = double{ order["price"] };
                const auto qty = double{ order["qty"] };
                const auto leaves_qty = double{ order["leaves_qty"] };
                const auto order_status = std::string_view{ order["order_status"] };
                std::string timestamp_str = std::string_view{ order["create_time"] }.data();
                if (timestamp_str.size() > 26) {
                    timestamp_str[26] = 'Z';
                }
                const auto timestamp = DateTime::iso8601_us_to_time_point_us(timestamp_str);
                if (order_status == "PartiallyFilled" || order_status == "PendingCancel") {
                    order_manager->portfolio->positions_buy[symbol.idx].qty += side == Side::buy ? qty : -qty;
                }
                if (order_status == "Created" || order_status == "New" || order_status == "PartiallyFilled" || order_status == "PendingCancel") {
                    order_manager->portfolio->update_order(id, symbol, side, leaves_qty, price, timestamp);
                }
                else if (order_status == "Rejected" || order_status == "Filled" || order_status == "Cancelled") {
                    order_manager->portfolio->remove_order(id);
                }
                order_manager->order_updated();
                logger.info("WS order: %s", message->c_str());
            }
        }
    }
    else if (topic == "execution") {
        for (auto order : doc["data"]) {
            const std::string_view id_str = order["order_link_id"];
            if (id_str.size() == 36) {
                const auto id = Uuid{ id_str };
                const auto symbol = string_to_symbol(order["symbol"]);
                const auto side = string_to_side(order["side"]);
                const auto price = double{ order["price"] };
                const auto qty = double{ order["leaves_qty"] };
                std::string timestamp_str = std::string_view{ order["trade_time"] }.data();
                if (timestamp_str.size() > 26) {
                    timestamp_str[26] = 'Z';
                }
                const auto timestamp = DateTime::iso8601_us_to_time_point_us(timestamp_str);
                order_manager->portfolio->update_order(id, symbol, side, qty, price, timestamp);
                order_manager->order_updated();
                logger.info("WS execution: %s", message->c_str());
            }
        }
    }
    else if (topic == "position") {
        for (auto order : doc["data"]) {
            const auto symbol = string_to_symbol(order["symbol"]);
            const auto qty = double{ order["size"] };
            const auto side = string_to_side(order["side"]);
            order_manager->portfolio->update_position(symbol, side, qty);
        }
        order_manager->position_updated();
    }
    else if (topic == "wallet") {
        for (auto balance : doc["data"]) {
            order_manager->portfolio->update_wallet(balance["wallet_balance"], balance["available_balance"]);
        }
    }
    else if (topic.starts_with("orderBookL2_25.")) {
        const auto& symbol = string_to_symbol(topic.substr(15));
        auto& order = (*order_manager->order_books)[symbol.idx];
        const std::string_view type = doc["type"];
        auto data = doc["data"];
        if (type == "snapshot") {
            for (auto entry : data["order_book"]) {
                const auto price = std::stod(std::string{ std::string_view{ entry["price"] } });
                const auto side = string_to_side(entry["side"]);
                const auto qty = double{ entry["size"] };
                order.insert(price, side, qty);
            }
        }
        else {
            for (auto entry : data["delete"]) {
                const auto price = std::stod(std::string{ std::string_view{ entry["price"] } });
                const auto side = string_to_side(entry["side"]);
                order.del(price, side);
            }
            for (auto entry : data["update"]) {
                const auto price = std::stod(std::string{ std::string_view{ entry["price"] } });
                const auto side = string_to_side(entry["side"]);
                const auto qty = double{ entry["size"] };
                order.update(price, side, qty);
            }
            for (auto entry : data["insert"]) {
                const auto price = std::stod(std::string{ std::string_view{ entry["price"] } });
                const auto side = string_to_side(entry["side"]);
                const auto qty = double{ entry["size"] };
                order.insert(price, side, qty);
            }
            order_manager->order_book_updated();
        }
    }
    else if (topic.starts_with("trade.")) {
        for (auto trade : doc["data"]) {
            const auto& symbol = string_to_symbol(trade["symbol"]);
            const auto price = std::stod(std::string{ std::string_view{ trade["price"] } });
            const auto side = string_to_side(trade["side"]);
            order_manager->portfolio->new_trade(symbol, side, price);
        }
    }
    else {
        auto a = 1;
    }
}

void ByBitWebSocket::on_resolve(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type results)
{
    //logger.info("ByBitWebSocket::on_resolve");

    if (ec) {
        fail(ec, "on_resolve");
        return;
    }

    boost::beast::get_lowest_layer(*websocket).expires_after(std::chrono::seconds{ 10 });
    boost::beast::get_lowest_layer(*websocket).async_connect(results, boost::beast::bind_front_handler(&ByBitWebSocket::on_connect, shared_from_this()));
}

void ByBitWebSocket::on_connect(boost::beast::error_code ec, boost::asio::ip::tcp::resolver::results_type::endpoint_type ep)
{
    //logger.info("ByBitWebSocket::on_connect");

    if (ec)
    {
        fail(ec, "on_connect");
        return;
    }

    // Update the host_ string. This will provide the value of the host HTTP header during the WebSocket handshake.
    // See https://tools.ietf.org/html/rfc7230#section-5.4
    host_address = std::string{ ByBit::WebSocket::host } + ":" + std::to_string(ep.port());

    // Set a timeout on the operation
    boost::beast::get_lowest_layer(*websocket).expires_after(std::chrono::seconds{ 30 });

    // Set SNI Hostname (many hosts need this to handshake successfully)
    if (!SSL_set_tlsext_host_name(
        websocket->next_layer().native_handle(),
        ByBit::WebSocket::host))
    {
        ec = boost::beast::error_code(static_cast<int>(::ERR_get_error()),
            boost::beast::net::error::get_ssl_category());
        return fail(ec, "connect");
    }

    // Perform the SSL handshake
    websocket->next_layer().async_handshake(boost::asio::ssl::stream_base::client, boost::beast::bind_front_handler(&ByBitWebSocket::on_ssl_handshake, shared_from_this()));
}

void ByBitWebSocket::on_ssl_handshake(boost::beast::error_code ec)
{
    //logger.info("ByBitWebSocket::on_ssl_handshake");

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
    websocket->async_handshake(host_address, url, boost::beast::bind_front_handler(&ByBitWebSocket::on_handshake, shared_from_this()));
}

void ByBitWebSocket::on_handshake(boost::beast::error_code ec)
{
    //logger.info("ByBitWebSocket::on_handshake");

    if (ec) {
        fail(ec, "on_handshake");
        return;
    }

    websocket_buffer.clear();
    websocket->async_read(websocket_buffer, boost::beast::bind_front_handler(&ByBitWebSocket::on_read, shared_from_this()));

    if (authenticate) {
        request_authentication();
    }
    else {
        subscribe();
    }
}

void ByBitWebSocket::on_write(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    //logger.info("ByBitWebSocket::on_write");

    if (ec) {
        return fail(ec, "write");
    }
}

void ByBitWebSocket::on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    //logger.info("ByBitWebSocket::on_read");
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

    // Get message
    auto ss = std::stringstream{};
    ss << boost::beast::make_printable(websocket_buffer.data());
    auto message = ss.str();

    // Prepare next message reception
    websocket_buffer.clear();
    websocket->async_read(websocket_buffer, boost::beast::bind_front_handler(&ByBitWebSocket::on_read, shared_from_this()));

    parse_message(&message);
}

void ByBitWebSocket::on_close(boost::beast::error_code ec)
{
    if (ec) {
        return fail(ec, "close");
    }
}

void ByBitWebSocket::websocket_worker(void)
{
    while (websocket_thread_running) {
        logger.info("ByBitWebSocket:websocket_worker: connect");
        connect();
        logger.info("ByBitWebSocket:websocket_worker: start");
        ioc.run();
        logger.info("ByBitWebSocket:websocket_worker: end");
    }
}

void ByBitWebSocket::heartbeat_worker(void)
{
    while (heartbeat_thread_running) {
        std::this_thread::sleep_for(30s);
        send_heartbeat();
    }
}
