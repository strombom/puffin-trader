#include "ByBitRest.h"
#include "ByBitConfig.h"
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

ByBitRest::ByBitRest(void) :
    user_order_id(0), connected(false), http2_thread_running(true), heartbeat_thread_running(true)
{
    heartbeat_reset();

    http2_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::http2_runner, this));
    heartbeat_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::heartbeat_runner, this));

    //std::this_thread::sleep_for(2s);
    //std::string method = "GET";
    //std::string uri = "https://api.bybit.com/public/linear/recent-trading-records?symbol=BTCUSDT&limit=1";
    //request(method, uri);
}

void ByBitRest::join(void)
{
    http2_thread->join();
}

bool ByBitRest::is_connected(void)
{
    return connected;
}

void ByBitRest::place_order(const std::string& symbol, double qty, double price)
{
    // {"api_key":"{api_key}","side"="Buy","symbol"="BTCUSD","order_type":"Market","qty":10,"time_in_force":"GoodTillCancel","timestamp":{timestamp},"sign":"{sign}"}

    std::string uri = "https://api-testnet.bybit.com/private/linear/order/create";


    //const auto expires = std::to_string(authenticator.generate_expiration(ByBit::websocket::auth_timeout));
    const auto timestamp = std::to_string(authenticator.generate_expiration(-1s));
    const auto sign_message = std::string{ "GET/realtime" } + timestamp;
    const auto signature = authenticator.authenticate(sign_message);

    //json11::Json auth_command = json11::Json::object{
    //    { "op", "auth" },
    //    { "args", json11::Json::array{ByBit::api_key, expires, signature} }
    //};

    auto data = std::string{ "{" };
    data += std::string{ "\"api_key\":\"" } + ByBit::api_key + "\"";
    data += ",\"side\":\"Buy\"";
    data += ",\"symbol\":\"" + symbol + "\"";
    data += ",\"order_type\":\"Limit\"";
    data += ",\"qty\":" + std::to_string(qty);
    data += ",\"price\":" + std::to_string(price);
    data += ",\"time_in_force\":\"GoodTillCancel\"";
    data += ",\"close_on_trigger\":\"false\"";
    data += ",\"reduce_only\":\"false\"";
    data += ",\"order_link_id\":\"" + std::to_string(user_order_id) + "\"";
    data += ",\"timestamp\":" + timestamp;
    data += ",\"sign\":\"" + signature + "\"";
    data += "}";    

    /*
    uri += "?side=Buy";
    uri += "&symbol=" + symbol;
    uri += "&order_type=Limit";
    uri += "&qty=" + std::to_string(qty);
    uri += "&price=" + std::to_string(price);
    uri += "&time_in_force=GoodTillCancel";
    uri += "&close_on_trigger=false";
    uri += "&reduce_only=false";
    uri += "&order_link_id=" + std::to_string(user_order_id);
    */

    logger.info("place_order: %s", data.c_str());

    post_request(uri, data);

    user_order_id++;
}

void ByBitRest::http2_runner(void)
{
    while (http2_thread_running) {

        //std::scoped_lock lock(connection_mutex);

        logger.info("http2_runner connecting");

        boost::system::error_code ec;

        tls_ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::sslv23);
        //boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
        tls_ctx->set_default_verify_paths();
        // tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer); // disabled to make development easier
        nghttp2::asio_http2::client::configure_tls_context(ec, *tls_ctx);

        io_service = std::make_unique<boost::asio::io_service>();
        session = std::make_unique<nghttp2::asio_http2::client::session>(*io_service, *tls_ctx, host, service);

        session->on_connect([this](tcp::resolver::iterator endpoint_it) {
            connected = true;
            logger.info("http2_runner connected: %s", (*endpoint_it).endpoint().data());
        });

        session->on_error([this](const boost::system::error_code& ec) {
            connected = false;
            logger.info("http2_runner error: %s", ec.message().c_str());
        });

        io_service->run();
        connected = false;
        session = nullptr;
        logger.info("http2_runner disconnected");

        //std::this_thread::sleep_for(5s);
    }
}

void ByBitRest::post_request(const std::string& uri, const std::string& data)
{
    logger.info("request start");

    if (!connected || session == nullptr) {
        return;
    }

    try {
        boost::system::error_code ec;
        //auto req = session->submit(ec, method, uri, { {"cookie", {}} });
        auto req = session->submit(ec, "POST", uri, data);
        if (ec) {
            logger.info("response error: %s", ec.message().data());
            return;
        }

        req->on_response([this](const nghttp2::asio_http2::client::response& res) {
            logger.info("response header was received");
            print_header(res);

            res.on_data([this](const uint8_t* data, std::size_t len) {
                this->on_data(reinterpret_cast<const char*>(data), len);
                });
            });

        req->on_close([](uint32_t error_code) {
            logger.info("request done with error_code: %d", error_code);
            });

        req->on_push([this](const nghttp2::asio_http2::client::request& push_req) {
            logger.info("request push request was received");

            print_header(push_req);

            push_req.on_response([this](const nghttp2::asio_http2::client::response& res) {
                logger.info("request push response header was received");

                res.on_data([this](const uint8_t* data, std::size_t len) {
                    this->on_data(reinterpret_cast<const char*>(data), len);
                    });
                });
            });

        heartbeat_reset();

    }
    catch (std::exception& e) {
        logger.info("request exception: %s", e.what());
    }
}

void ByBitRest::get_request(const std::string& uri)
{
    logger.info("request start");

    //std::string uri = "https://api.bybit.com/public/linear/kline?symbol=BTCUSDT&interval=1&limit=2&from=1581231260";
    //std::string uri = "https://api.bybit.com/public/linear/recent-trading-records?symbol=BTCUSDT&limit=1";
    //std::string scheme, host, service;

    //if (nghttp2::asio_http2::host_service_from_uri(ec, scheme, host, service, uri)) {
    //    std::cerr << "error: bad URI: " << ec.message() << std::endl;
    //    return;
    //}

    //boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
    //tls_ctx.set_default_verify_paths();
    // disabled to make development easier...
    // tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer);
    //nghttp2::asio_http2::client::configure_tls_context(ec, tls_ctx);

    if (!connected || session == nullptr) {
        return;
    }

    try {
        boost::system::error_code ec;
        //auto req = session->submit(ec, method, uri, { {"cookie", {}} });
        auto req = session->submit(ec, "GET", uri);
        if (ec) {
            logger.info("response error: %s", ec.message().data());
            return;
        }

        req->on_response([this](const nghttp2::asio_http2::client::response& res) {
            logger.info("response header was received");
            print_header(res);

            res.on_data([this](const uint8_t* data, std::size_t len) {
                this->on_data(reinterpret_cast<const char*>(data), len);
            });
        });

        req->on_close([](uint32_t error_code) {
            logger.info("request done with error_code: %d", error_code);
        });

        req->on_push([this](const nghttp2::asio_http2::client::request& push_req) {
            logger.info("request push request was received");

            print_header(push_req);

            push_req.on_response([this](const nghttp2::asio_http2::client::response& res) {
                logger.info("request push response header was received");

                res.on_data([this](const uint8_t* data, std::size_t len) {
                    this->on_data(reinterpret_cast<const char*>(data), len);
                });
            });
        });

        heartbeat_reset();

    } catch (std::exception& e) {
        logger.info("request exception: %s", e.what());
    }
}

void ByBitRest::on_data(const char* data, std::size_t len)
{
    logger.info("on_data: %s", data);
    //std::cerr.write(reinterpret_cast<const char*>(data), len);
    //std::cerr << std::endl;
}

void ByBitRest::heartbeat_runner(void)
{
    while (heartbeat_thread_running) {
        std::this_thread::sleep_for(1s);
        if (DateTime::now() > heartbeat_timeout) {
            logger.info("heartbeat request");
            get_request("https://api.bybit.com/public/linear/recent-trading-records?symbol=BTCUSDT&limit=1");
        }
    }
}

void ByBitRest::heartbeat_reset(void)
{
    logger.info("heartbeat reset");
    heartbeat_timeout = DateTime::now() + 6s;
}
