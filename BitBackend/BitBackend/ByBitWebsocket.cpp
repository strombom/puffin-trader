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

    auto a = 1;


    //auto error_message = std::string{ "{\"command\":\"error\"}" };
    //const auto command = json11::Json::parse(message.c_str(), error_message);
    /*
    if (authenticate && command["request"]["op"] == "auth") {
        if (command["success"].bool_value()) {
            logger.info("Authenticated");
            subscribe();
        }
        else {
            request_authentication();
        }
    }
    else if (command["topic"] == "wallet") {
        for (const auto& data : command["data"].array_items()) {
            order_manager->portfolio->update_wallet(data["wallet_balance"].number_value(), data["available_balance"].number_value());
        }
    }
    else if (command["topic"] == "order") {
        for (const auto& data : command["data"].array_items()) {

            const auto id_str = data["order_link_id"].string_value();
            if (id_str.size() == 36) {
                auto timestamp_string = data["create_time"].string_value();
                if (timestamp_string.size() > 26) {
                    timestamp_string[26] = 'Z'; // Remove nanosecond part, DateTime can only parse microseconds
                }
                const auto timestamp = DateTime::iso8601_us_to_time_point_us(timestamp_string);
                const auto id = Uuid{ id_str };
                const auto symbol = string_to_symbol(data["symbol"].string_value());
                const auto side = data["side"].string_value() == "Buy" ? Side::buy : Side::sell;
                const auto price = data["price"].number_value();
                const auto qty = data["leaves_qty"].number_value();
                const auto confirmed = true;
                order_manager->portfolio->update_order(id, symbol, side, qty, price, timestamp, confirmed);
                order_manager->order_updated();
            }
            else {
                logger.info("order update invalid %s", data["order_id"].string_value().c_str());
            }
        }
        //order_manager->portfolio_updated();
    }
    else if (command["topic"] == "execution") {
        for (const auto& data : command["data"].array_items()) {
            const auto id_str = data["order_link_id"].string_value();
            if (id_str.size() == 36) {
                auto timestamp_string = data["trade_time"].string_value();
                if (timestamp_string.size() > 26) {
                    timestamp_string[26] = 'Z'; // Remove nanosecond part, DateTime can only parse microseconds
                }
                const auto timestamp = DateTime::iso8601_us_to_time_point_us(timestamp_string);
                const auto id = Uuid{ id_str };
                const auto symbol = string_to_symbol(data["symbol"].string_value());
                const auto side = data["side"].string_value() == "Buy" ? Side::buy : Side::sell;
                const auto price = data["price"].number_value();
                const auto qty = data["leaves_qty"].number_value();
                const auto confirmed = true;
                 order_manager->portfolio->update_order(id, symbol, side, qty, price, timestamp, confirmed);
                 order_manager->order_updated();
            }
            else {
                logger.info("order execution invalid %s", data["order_id"].string_value().c_str());
            }
        }
        //logger.info("Execution: %s", message.c_str());
    }
    else if (command["topic"] == "position") {
        for (const auto& data : command["data"].array_items()) {
            const auto& symbol = string_to_symbol(data["symbol"].string_value());
            const auto side = data["side"].string_value() == "Buy" ? Side::buy : Side::sell;
            const auto qty = data["size"].number_value();
            order_manager->portfolio->update_position(symbol, side, qty);
        }
        order_manager->portfolio_updated();
    }
    else if (command["ret_msg"] == "pong") {
        // Example: {"success":true,"ret_msg":"pong","conn_id":"bc172b63-001d-47b2-b9e1-37ce4f0264ce","request":{"op":"ping","args":null}}
        if (command["success"].bool_value()) {
            //logger.info("Pong: success");
        }
        else {
            logger.info("Pong: fail");
        }
    }
    else if (command["topic"].string_value().starts_with("trade.")) {
        if (command["data"].is_array() && command["data"].array_items().size() > 0) {
            const auto& data = command["data"].array_items().back();
            const auto& symbol = string_to_symbol(command["topic"].string_value().substr(6));
            //const auto& tick_direction = command["tick_direction"].string_value();
            // Tick direction: PlusTick, ZeroPlusTick, MinusTick, ZeroMinusTick
            //const auto side = tick_direction[0] == 'P' || tick_direction[0] == 'Z' ? Portfolio::Side::buy : Portfolio::Side::sell;
            const auto price = std::stod(data["price"].string_value());
            const auto side = data["side"].string_value() == "Buy" ? Side::buy : Side::sell;
            order_manager->portfolio->new_trade(symbol, side, price);
        }
    }
    else if (command["topic"].string_value().starts_with("orderBookL2_25.")) {
        const auto& symbol = string_to_symbol(command["topic"].string_value().substr(15));
        if (command["type"] == "snapshot") {
            (*order_manager->order_books)[symbol.idx].clear();
            for (const auto& data : command["data"]["order_book"].array_items()) {
                (*order_manager->order_books)[symbol.idx].insert(
                    std::stod(data["price"].string_value()),
                    data["side"].string_value() == "Buy" ? Side::buy : Side::sell,
                    data["size"].number_value()
                );
            }
        }
        else if (command["type"] == "delta") {
            for (const auto& data : command["data"]["delete"].array_items()) {
                (*order_manager->order_books)[symbol.idx].del(
                    std::stod(data["price"].string_value()),
                    data["side"].string_value() == "Buy" ? Side::buy : Side::sell
                );
            }
            for (const auto& data : command["data"]["update"].array_items()) {
                (*order_manager->order_books)[symbol.idx].update(
                    std::stod(data["price"].string_value()),
                    data["side"].string_value() == "Buy" ? Side::buy : Side::sell,
                    data["size"].number_value()
                );
            }
            for (const auto& data : command["data"]["insert"].array_items()) {
                (*order_manager->order_books)[symbol.idx].insert(
                    std::stod(data["price"].string_value()),
                    data["side"].string_value() == "Buy" ? Side::buy : Side::sell,
                    data["size"].number_value()
                );
            }
            order_manager->order_book_updated();
            //order_manager->tick();
            //logger.info("Bid %.2f", (*order_books)[symbol.idx].get_last_bid());
            //(*order_manager->order_books)[symbol.idx].updated();
        }
    }
    else {
        logger.info("Other: %s", message.c_str());
    }
    */

    /*
    else if (command["info"].string_value() == "Welcome to the BitMEX Realtime API.") {
        request_authentication();
    }
    else if (command["request"]["op"].string_value() == "authKeyExpires") {
        if (command["success"].bool_value() == true) {
            json11::Json subscribe_command = json11::Json::object{
                { "op", "subscribe" },
                { "args", json11::Json::array{"order", "position", "margin", "wallet", "trade:XBTUSD", "orderBook10:XBTUSD"} }
            };
            send(subscribe_command.dump());
        }
        else {
            std::this_thread::sleep_for(2s);
            request_authentication();
        }
    }
    else if (command["table"].string_value() == "order") {
        //logger.info("BitmexWebSocket::parse_message: order (%s)", message.c_str());

        const auto& action = command["action"].string_value();

        for (const auto& data : command["data"].array_items()) {
            const auto order_id = data["orderID"].string_value();
            const auto symbol = data["symbol"].string_value();
            const auto timestamp = DateTime::to_time_point_ms(data["timestamp"].string_value(), "%FT%TZ");

            if ((action == "insert" || action == "partial") && (data["ordStatus"] == "New" || data["ordStatus"] == "Partially filled")) {
                const auto buy = (data["side"].string_value() == "Buy");
                const auto order_size = data["orderQty"].int_value();
                const auto price = data["price"].number_value();
                //bitmex_account->insert_order(symbol, order_id, timestamp, buy, order_size, price);
            }
            else if ((action == "insert" || action == "partial") && data["ordStatus"] == "Canceled") {
                // Ignore
            }
            else if (action == "update" && (data["ordStatus"].string_value() == "Filled" || data["ordStatus"].string_value() == "Partially filled")) {
                const auto remaining_size = data["leavesQty"].int_value();
                //bitmex_account->fill_order(symbol, order_id, timestamp, remaining_size);
            }
            else if (action == "update" && data["ordStatus"].string_value() == "Canceled") {
                //bitmex_account->delete_order(order_id);
            }
            else if (action == "update" && data["price"].is_number() && data["orderQty"].is_number()) {
                const auto price = data["price"].number_value();
                const auto size = data["orderQty"].int_value();
                //bitmex_account->amend_order(symbol, order_id, timestamp, size, price);
            }
            else if (action == "update" && data["price"].is_number()) {
                const auto price = data["price"].number_value();
                //bitmex_account->amend_order_price(symbol, order_id, timestamp, price);
            }
            else if (action == "update" && data["orderQty"].is_number()) {
                const auto size = data["orderQty"].int_value();
                //bitmex_account->amend_order_size(symbol, order_id, timestamp, size);
            }
            else if (action == "update") {

            }
            else  if (action == "delete") {
                //bitmex_account->delete_order(order_id);
            }
            else {
                logger.info("ByBitWebSocket::parse_message: order unknown (%s)", message.c_str());
            }
        }
    }
    else if (command["table"].string_value() == "trade") {
        for (const auto& data : command["data"].array_items()) {
            const auto& symbol = data["symbol"].string_value();
            if (symbol == "XBTUSD") {
                const auto price = data["price"].number_value();
                //bitmex_account->set_mark_price(price);
            }
        }
    }
    else if (command["table"].string_value() == "position") {
        for (const auto& data : command["data"].array_items()) {
            if (data["markValue"].is_number()) {
                const auto mark_value = data["markValue"].number_value();
                //bitmex_account->set_leverage(mark_value);
            }
            if (data["unrealisedPnl"].is_number()) {
                const auto upnl = data["unrealisedPnl"].number_value() / 100000000.0; // Convert from Satoshis to Bitcoin
                //bitmex_account->set_upnl(upnl);
            }
            if (data["currentQty"].is_number()) {
                const auto contracts = (int)data["currentQty"].number_value();
                //bitmex_account->set_contracts(contracts);
            }
        }
    }
    else if (command["table"].string_value() == "wallet") {
        for (const auto& data : command["data"].array_items()) {
            const auto amount = data["amount"].number_value() / 100000000.0; // Convert from Satoshis to Bitcoin
            //bitmex_account->set_wallet(amount);
        }
    }
    else if (command["table"].string_value() == "orderBook10") {
        for (const auto& data : command["data"].array_items()) {
            const auto& symbol = data["symbol"].string_value();
            if (symbol == "XBTUSD") {
                const auto ask_price = data["asks"].array_items()[0][0].number_value();
                const auto bid_price = data["bids"].array_items()[0][0].number_value();
                //bitmex_account->set_ask_price(ask_price);
                //bitmex_account->set_bid_price(bid_price);
            }
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
        logger.info("ByBitWebSocket::parse_message: unknown command (%s)", message.c_str());
    }
    */
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
