#include "ByBitRest.h"
#include "BitLib/Logger.h"


using boost::asio::ip::tcp;
using namespace std::chrono_literals;


void print_header(const nghttp2::asio_http2::header_map& h) {
    for (auto& kv : h) {
        std::cerr << kv.first << ": " << kv.second.value << "\n";
    }
    std::cerr << std::endl;
}

void print_header(const nghttp2::asio_http2::client::response& res) {
    std::cerr << "HTTP/2 " << res.status_code() << "\n";
    print_header(res.header());
}

void print_header(const nghttp2::asio_http2::client::request& req) {
    auto& uri = req.uri();
    std::cerr << req.method() << " " << uri.scheme << "://" << uri.host
        << uri.path;
    if (!uri.raw_query.empty()) {
        std::cerr << "?" << uri.raw_query;
    }
    std::cerr << " HTTP/2\n";
    print_header(req.header());
}

ByBitRest::ByBitRest(sptrOrderManager order_manager) :
    order_manager(order_manager), connected(false), http2_thread_running(true), heartbeat_thread_running(true)
{
    using namespace std::placeholders;
    order_manager->set_callbacks(
        std::bind(&ByBitRest::place_order, this, _1, _2, _3, _4), 
        std::bind(&ByBitRest::cancel_order, this, _1, _2),
        std::bind(&ByBitRest::replace_order, this, _1, _2, _3, _4)
    );

    heartbeat_reset();

    http2_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::http2_runner, this));
    heartbeat_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::heartbeat_runner, this));

    while (!is_connected()) {
        std::this_thread::sleep_for(100ms);
    }
}

void ByBitRest::join(void)
{
    http2_thread->join();
    heartbeat_thread->join();
}

bool ByBitRest::is_connected(void)
{
    return connected;
}

void ByBitRest::place_order(const Symbol& symbol, Side side, double qty, double price)
{
    const auto id = uuid_generator.generate();
    const auto timestamp_sign = std::to_string(authenticator.generate_expiration(-2s));

    auto price_str = std::to_string(price);
    price_str.erase(price_str.find_last_not_of('0') + 1, std::string::npos);
    price_str.erase(price_str.find_last_not_of('.') + 1, std::string::npos);

    auto qty_str = std::to_string(std::abs(qty));
    qty_str.erase(qty_str.find_last_not_of('0') + 1, std::string::npos);
    qty_str.erase(qty_str.find_last_not_of('.') + 1, std::string::npos);

    const auto reduce_only_str = qty > 0 ? std::string{ "false" } : std::string{ "true" };

    const auto side_str = side == Side::buy ? std::string{ "Buy" } : std::string{ "Sell" };

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&close_on_trigger=false";
    sign_message += "&order_link_id=" + id.to_string();
    sign_message += "&order_type=Limit";
    sign_message += "&price=" + price_str;
    sign_message += "&qty=" + qty_str;
    sign_message += "&reduce_only=" + reduce_only_str;
    sign_message += "&side=" + side_str;
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&time_in_force=PostOnly";
    sign_message += "&timestamp=" + timestamp_sign;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"close_on_trigger\":false";
    data += ",\"order_link_id\":\"" + id.to_string() + "\"";
    data += ",\"order_type\":\"Limit\"";
    data += ",\"price\":" + price_str;
    data += ",\"qty\":" + qty_str;
    data += ",\"reduce_only\":" + reduce_only_str;
    data += ",\"side\":\"" + side_str + "\"";
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"time_in_force\":\"PostOnly\"";
    data += ",\"timestamp\":" + timestamp_sign;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    //logger.info("place_order: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::create_order, id);

    //const auto side = qty > 0 ? Side::buy : Side::sell;
    const auto timestamp = DateTime::now();
    const bool confirmed = false;
    order_manager->portfolio->update_order(id, symbol, side, std::abs(qty), price, timestamp, confirmed);
}

void ByBitRest::replace_order(const Symbol& symbol, Uuid id_external, double qty, double price)
{
    const auto timestamp_sign = std::to_string(authenticator.generate_expiration(-2s));

    auto qty_str = std::to_string(qty);
    qty_str.erase(qty_str.find_last_not_of('0') + 1, std::string::npos);
    qty_str.erase(qty_str.find_last_not_of('.') + 1, std::string::npos);

    auto price_str = std::to_string(price);
    price_str.erase(price_str.find_last_not_of('0') + 1, std::string::npos);
    price_str.erase(price_str.find_last_not_of('.') + 1, std::string::npos);

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&order_link_id=" + id_external.to_string();
    if (qty > 0) {
        sign_message += "&p_r_qty=" + qty_str;
    }
    else {
        sign_message += "&p_r_price=" + price_str;
    }
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&timestamp=" + timestamp_sign;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"order_link_id\":\"" + id_external.to_string() + "\"";
    if (qty > 0) {
        data += ",\"p_r_qty\":" + qty_str;
    }
    else {
        data += ",\"p_r_price\":" + price_str;
    }
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"timestamp\":" + timestamp_sign;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    //logger.info("replace_order: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::replace_order, id_external);

    //const auto side = qty > 0 ? Side::buy : Side::sell;
    //const auto timestamp = DateTime::now();
    //const bool confirmed = false;
    order_manager->portfolio->replace_order(id_external);
    //order_manager->portfolio->update_order(id.to_string(), symbol, side, std::abs(qty), price, timestamp, confirmed);
}

void ByBitRest::cancel_all_orders(const Symbol& symbol)
{
    const auto timestamp = std::to_string(authenticator.generate_expiration(-2s));

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&timestamp=" + timestamp;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"timestamp\":" + timestamp;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    //logger.info("cancel_all_orders: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::cancel_all_orders, Uuid{});
}

void ByBitRest::cancel_order(const Symbol& symbol, Uuid id_external)
{
    const auto timestamp = std::to_string(authenticator.generate_expiration(-2s));

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&order_link_id=" + id_external.to_string();
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&timestamp=" + timestamp;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"order_link_id\":\"" + id_external.to_string() + "\"";
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"timestamp\":" + timestamp;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    //logger.info("cancel_order: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::cancel_order, id_external);
}

void ByBitRest::get_position(const Symbol& symbol)
{
    const auto timestamp = std::to_string(authenticator.generate_expiration(-2s));

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&timestamp=" + timestamp;
    auto signature = authenticator.authenticate(sign_message);

    auto query = std::string{};
    query += "?api_key=" + std::string{ ByBit::api_key };
    query += "&symbol=" + std::string{ symbol.name };
    query += "&timestamp=" + timestamp;
    query += "&sign=" + signature;

    //logger.info("get_positions: %s", query.c_str());
    get_request(query, ByBit::Rest::Endpoint::position_list, Uuid{});
}

void ByBitRest::http2_runner(void)
{
    while (http2_thread_running) {
        logger.info("http2_runner connecting");

        boost::system::error_code ec;

        tls_ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::sslv23);
        //boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
        tls_ctx->set_default_verify_paths();
        //tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer); // disabled to make development easier
        nghttp2::asio_http2::client::configure_tls_context(ec, *tls_ctx);

        io_service = std::make_unique<boost::asio::io_service>();
        session = std::make_unique<nghttp2::asio_http2::client::session>(*io_service, *tls_ctx, ByBit::Rest::host, ByBit::Rest::service);

        session->on_connect([this](tcp::resolver::iterator endpoint_it) {
            connected = true;
            logger.info("http2_runner connected");
        });

        session->on_error([this](const boost::system::error_code& ec) {
            connected = false;
            logger.info("http2_runner error: %s", ec.message().c_str());
        });

        /*
        const auto expires = std::to_string(authenticator.generate_expiration(ByBit::websocket::auth_timeout));
        const auto sign_message = std::string{ "GET/realtime" } + expires;
        const auto signature = authenticator.authenticate(sign_message);

        json11::Json auth_command = json11::Json::object{
            { "op", "auth" },
            { "args", json11::Json::array{ByBit::api_key, expires, signature} }
        };

        std::string uri = "https://api2-testnet.bybit.com/private/linear/order/create";
        post_request(uri, auth_command.dump());
        */

        //if (nghttp2::asio_http2::host_service_from_uri(ec, scheme, host, service, uri)) {
        //    std::cerr << "error: bad URI: " << ec.message() << std::endl;
        //    return;
        //}

        //boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
        //tls_ctx.set_default_verify_paths();
        // disabled to make development easier...
        // tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer);
        //nghttp2::asio_http2::client::configure_tls_context(ec, tls_ctx);

        io_service->run();
        connected = false;
        session = nullptr;
        logger.info("http2_runner disconnected");

        //std::this_thread::sleep_for(5s);
    }
}

void ByBitRest::post_request(const std::string& data, ByBit::Rest::Endpoint endpoint, Uuid id)
{
    //logger.info("post_request start");

    if (!connected || session == nullptr) {
        return;
    }

    try {
        boost::system::error_code ec;
        const auto uri = std::string{ ByBit::Rest::base_endpoint } + std::string{ ByBit::Rest::endpoints[(int)endpoint] };
        auto req = session->submit(ec, "POST", uri, data, { {"Content-Type", {"application/json"}} });
        if (ec) {
            logger.info("response error: %s", ec.message().data());
            return;
        }

        auto buffer = std::make_shared<std::string>();
        req->on_response([this, endpoint, buffer](const nghttp2::asio_http2::client::response& res) {
            //logger.info("response header was received");
            //print_header(res);

            res.on_data([this, endpoint, buffer](const uint8_t* data, std::size_t len) {
                //this->on_data(reinterpret_cast<const char*>(data), len, endpoint);
                *buffer += std::string{ (const char*)data, len };
            });
        });

        req->on_close([this, endpoint, buffer, id](uint32_t error_code) {
            if (error_code == 0) {
                this->on_data(reinterpret_cast<const char*>(buffer->c_str()), buffer->size(), endpoint, id);
            }
            else {
                logger.info("request done with error_code: %d", error_code);
            }
        });

        heartbeat_reset();

    }
    catch (std::exception& e) {
        logger.info("post_request exception: %s", e.what());
    }
}

void ByBitRest::get_request(const std::string& query, ByBit::Rest::Endpoint endpoint, Uuid id)
{
    //logger.info("get_request start");

    if (!connected || session == nullptr) {
        return;
    }

    try {
        boost::system::error_code ec;
        const auto uri = std::string{ ByBit::Rest::base_endpoint } + ByBit::Rest::endpoints[(int)endpoint] + query;
        auto req = session->submit(ec, "GET", uri);
        if (ec) {
            logger.info("response error: %s", ec.message().data());
            return;
        }

        auto buffer = std::make_shared<std::string>();
        req->on_response([this, endpoint, buffer](const nghttp2::asio_http2::client::response& res) {
            //logger.info("response header was received");
            //print_header(res);

            res.on_data([this, endpoint, buffer](const uint8_t* data, std::size_t len) {
                //this->on_data(reinterpret_cast<const char*>(data), len, endpoint);
                *buffer += std::string{ (const char*)data, len };
            });
        });

        req->on_close([this, endpoint, buffer, id](uint32_t error_code) {
            if (error_code == 0) {
                this->on_data(reinterpret_cast<const char*>(buffer->c_str()), buffer->size(), endpoint, id);
            }
            else {
                logger.info("request done with error_code: %d", error_code);
            }
        });

        heartbeat_reset();

    } catch (std::exception& e) {
        logger.info("get_request exception: %s", e.what());
    }
}

void ByBitRest::on_data(const char* data, std::size_t len, ByBit::Rest::Endpoint endpoint, const Uuid& id)
{
    if (endpoint == ByBit::Rest::Endpoint::heartbeat_ping) {
        return;
    }

    auto str = std::string{ data, len };
    str.reserve(str.size() + simdjson::SIMDJSON_PADDING);
    auto doc = json_parser.iterate(str);

    long ret_code = doc["ret_code"];
    if (ret_code != 0) {
        if (ret_code == 20001 && !id.is_null()) {
            // order not exists or too late to cancel
            order_manager->portfolio->remove_order(id);
            order_manager->order_updated();
            logger.info("on_data, order does not exist %s", id.to_string().c_str());
        }
        if (ret_code == 130125 && !id.is_null()) {
            // current position is zero, cannot fix reduce-only order qty
            order_manager->portfolio->remove_order(id);
            order_manager->order_updated();
            logger.info("on_data, position is zero, cannot place reduce-only order %s", id.to_string().c_str());
        }
        else {
            std::string_view ret_msg = doc["ret_msg"]; // .find_field("ret_msg");
            logger.info("on_data, ret_code: %d %d %s %s", endpoint, ret_code, ret_msg.data(), std::string{ data, len }.c_str());
            return;
        }
    }
    std::string_view ret_msg = doc["ret_msg"]; // .find_field("ret_msg");
    if (ret_msg != "OK") {
        logger.info("on_data, ret_msg: %d %s %s", endpoint, ret_msg.data(), std::string{ data, len }.c_str());
        return;
    }
    std::string_view ext_code = doc["ext_code"]; // .find_field("ext_code");
    if (ext_code != "") {
        logger.info("on_data, ext_code: %d %s %s", endpoint, ext_code.data(), std::string{ data, len }.c_str());
        return;
    }
    std::string_view ext_info = doc["ext_info"]; // .find_field("ext_code");
    if (ext_code != "") {
        logger.info("on_data, ext_info: %d %s %s", endpoint, ext_info.data(), std::string{ data, len }.c_str());
        return;
    }

    if (endpoint == ByBit::Rest::Endpoint::position_list) {
        for (auto pos : doc["result"]) {
            auto symbol = string_to_symbol(pos["symbol"]);
            auto side = string_to_side(pos["side"]);
            double qty = pos["size"];

            order_manager->portfolio->update_position(symbol, side, qty);
        }
        order_manager->portfolio_updated();
    }
    else if (endpoint == ByBit::Rest::Endpoint::cancel_all_orders) {
        logger.info("on_data, cancel_all_orders OK");
    }
    else if (endpoint == ByBit::Rest::Endpoint::create_order) {
        const auto symbol = string_to_symbol(doc["result"]["symbol"]);
        const auto side = string_to_side(doc["result"]["side"]);
        const double price = doc["result"]["price"];
        const double qty = doc["result"]["qty"];
        const auto id = Uuid{ std::string_view{doc["result"]["order_link_id"]} };
        const auto timestamp = DateTime::iso8601_us_to_time_point_us(std::string_view{ doc["result"]["updated_time"] });
        const bool confirmed = true;
        order_manager->portfolio->update_order(id, symbol, side, qty, price, timestamp, confirmed);
        order_manager->order_updated();
        //logger.info("on_data: %d %d %s", endpoint, len, str.c_str());
    }
    else if (endpoint == ByBit::Rest::Endpoint::replace_order) {
        logger.info("on_data replace_order: %d %d %s", endpoint, len, str.c_str());
    }
    else {
        logger.info("on_data: %d %d %s", endpoint, len, str.c_str());
    }
}

void ByBitRest::heartbeat_runner(void)
{
    while (heartbeat_thread_running) {
        std::this_thread::sleep_for(1s);
        if (DateTime::now() > heartbeat_timeout) {
            get_request("?symbol=BTCUSDT&limit=1", ByBit::Rest::Endpoint::heartbeat_ping, Uuid{});
        }
    }
}

void ByBitRest::heartbeat_reset(void)
{
    heartbeat_timeout = DateTime::now() + 6s;
}
