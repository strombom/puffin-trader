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
    user_order_id(3), order_manager(order_manager), connected(false), http2_thread_running(true), heartbeat_thread_running(true)
{
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

int ByBitRest::place_order(const Symbol& symbol, double qty, double price)
{
    const auto timestamp = std::to_string(authenticator.generate_expiration(-2s));

    auto price_str = std::to_string(price);
    price_str.erase(price_str.find_last_not_of('0') + 1, std::string::npos);
    price_str.erase(price_str.find_last_not_of('.') + 1, std::string::npos);

    auto qty_str = std::to_string(qty);
    qty_str.erase(qty_str.find_last_not_of('0') + 1, std::string::npos);
    qty_str.erase(qty_str.find_last_not_of('.') + 1, std::string::npos);

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&close_on_trigger=false";
    sign_message += "&order_link_id=" + std::to_string(user_order_id);
    sign_message += "&order_type=Limit";
    sign_message += "&price=" + price_str;
    sign_message += "&qty=" + qty_str;
    sign_message += "&reduce_only=false";
    sign_message += "&side=Buy";
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&time_in_force=PostOnly";
    sign_message += "&timestamp=" + timestamp;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"close_on_trigger\":false";
    data += ",\"order_link_id\":\"" + std::to_string(user_order_id) + "\"";
    data += ",\"order_type\":\"Limit\"";
    data += ",\"price\":" + price_str + "";
    data += ",\"qty\":" + qty_str + "";
    data += ",\"reduce_only\":false";
    data += ",\"side\":\"Buy\"";
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"time_in_force\":\"PostOnly\"";
    data += ",\"timestamp\":" + timestamp;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    logger.info("place_order: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::create_order);

    return user_order_id++;
}

void ByBitRest::cancel_order(const Symbol& symbol, int _user_order_id)
{
    const auto timestamp = std::to_string(authenticator.generate_expiration(-2s));

    auto sign_message = std::string{};
    sign_message += "api_key=" + std::string{ ByBit::api_key };
    sign_message += "&order_link_id=" + std::to_string(user_order_id);
    sign_message += "&symbol=" + std::string{ symbol.name };
    sign_message += "&timestamp=" + timestamp;
    auto signature = authenticator.authenticate(sign_message);

    auto data = std::string{ "{" };
    data += "\"api_key\":\"" + std::string{ ByBit::api_key } + "\"";
    data += ",\"order_link_id\":\"" + std::to_string(user_order_id) + "\"";
    data += ",\"symbol\":\"" + std::string{ symbol.name } + "\"";
    data += ",\"timestamp\":" + timestamp;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";

    logger.info("cancel_order: %s", data.c_str());
    post_request(data, ByBit::Rest::Endpoint::cancel_order);
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

    logger.info("get_positions: %s", query.c_str());
    get_request(query, ByBit::Rest::Endpoint::position_list);
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

void ByBitRest::post_request(const std::string& data, ByBit::Rest::Endpoint endpoint)
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

        req->on_close([this, endpoint, buffer](uint32_t error_code) {
            if (error_code == 0) {
                this->on_data(reinterpret_cast<const char*>(buffer->c_str()), buffer->size(), endpoint);
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

void ByBitRest::get_request(const std::string& query, ByBit::Rest::Endpoint endpoint)
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

        req->on_close([this, endpoint, buffer](uint32_t error_code) {
            if (error_code == 0) {
                this->on_data(reinterpret_cast<const char*>(buffer->c_str()), buffer->size(), endpoint);
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

void ByBitRest::on_data(const char* data, std::size_t len, ByBit::Rest::Endpoint endpoint)
{
    auto str = std::string{ data, len };
    str.reserve(str.size() + simdjson::SIMDJSON_PADDING);
    auto doc = json_parser.iterate(str);

    logger.info("on_data: %d %d %s", endpoint, len, str.c_str());

    long ret_code = doc["ret_code"];
    if (ret_code != 0) {
        logger.info("on_data, ret_code: %d", ret_code);
        return;
    }
    std::string_view ret_msg = doc["ret_msg"]; // .find_field("ret_msg");
    if (ret_msg != "OK") {
        logger.info("on_data, ret_msg: %s", ret_msg.data());
        return;
    }
    std::string_view ext_code = doc["ext_code"]; // .find_field("ext_code");
    if (ext_code != "") {
        logger.info("on_data, ext_code: %s", ext_code.data());
        return;
    }
    std::string_view ext_info = doc["ext_info"]; // .find_field("ext_code");
    if (ext_code != "") {
        logger.info("on_data, ext_info: %s", ext_info.data());
        return;
    }

    if (endpoint == ByBit::Rest::Endpoint::position_list) {
        for (auto pos : doc["result"]) {
            auto symbol = string_to_symbol(pos["symbol"]);
            auto side = string_to_side(pos["side"]);
            double qty = pos["size"];

            order_manager->portfolio->update_position(symbol, side, qty);
        }
    }
}

void ByBitRest::heartbeat_runner(void)
{
    while (heartbeat_thread_running) {
        std::this_thread::sleep_for(1s);
        if (DateTime::now() > heartbeat_timeout) {
            get_request("?symbol=BTCUSDT&limit=1", ByBit::Rest::Endpoint::heartbeat_ping);
        }
    }
}

void ByBitRest::heartbeat_reset(void)
{
    //logger.info("heartbeat reset");
    heartbeat_timeout = DateTime::now() + 6s;
}
