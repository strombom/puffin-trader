#include "ByBitRest.h"


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
    connected(false), http2_thread_running(true), heartbeat_thread_running(true)
{
    heartbeat_reset();

    http2_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::http2_runner, this));
    heartbeat_thread = std::make_unique<std::thread>(std::bind(&ByBitRest::heartbeat_runner, this));

    //std::this_thread::sleep_for(2s);
    //std::string method = "GET";
    //std::string uri = "https://api.bybit.com/public/linear/recent-trading-records?symbol=BTCUSDT&limit=1";
    //request(method, uri);

    http2_thread->join();
}

void ByBitRest::http2_runner(void)
{
    while (http2_thread_running) {

        boost::system::error_code ec;

        tls_ctx = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::sslv23);
        //boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
        tls_ctx->set_default_verify_paths();
        // tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer); // disabled to make development easier
        nghttp2::asio_http2::client::configure_tls_context(ec, *tls_ctx);

        session = std::make_unique<nghttp2::asio_http2::client::session>(io_service, *tls_ctx, host, service);

        session->on_connect([&](tcp::resolver::iterator endpoint_it) {
            std::cerr << "connected to " << (*endpoint_it).endpoint() << std::endl;
            boost::system::error_code ec;
        });

        session->on_error([](const boost::system::error_code& ec) {
            std::cerr << "error: " << ec.message() << std::endl;
        });

        io_service.run();
    }
}

void ByBitRest::request(const std::string& method, const std::string& uri)
{
    heartbeat_reset();
    std::cerr << "request" << std::endl;

    boost::system::error_code ec;

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

    auto req = session->submit(ec, method, uri, { {"cookie", {}} });
    if (ec) {
        std::cerr << "error: " << ec.message() << std::endl;
        return;
    }

    req->on_response([this](const nghttp2::asio_http2::client::response& res) {
        std::cerr << "response header was received" << std::endl;
        print_header(res);

        res.on_data([this](const uint8_t* data, std::size_t len) {
            this->on_data(reinterpret_cast<const char*>(data), len);
        });
    });

    req->on_close([](uint32_t error_code) {
        std::cerr << "request done with error_code=" << error_code << std::endl;
    });

    req->on_push([this](const nghttp2::asio_http2::client::request& push_req) {
        std::cerr << "push request was received" << std::endl;

        print_header(push_req);

        push_req.on_response([this](const nghttp2::asio_http2::client::response& res) {
            std::cerr << "push response header was received" << std::endl;

            res.on_data([this](const uint8_t* data, std::size_t len) {
                this->on_data(reinterpret_cast<const char*>(data), len);
            });
        });
    });
}

void ByBitRest::on_data(const char* data, std::size_t len)
{
    std::cerr << "on_data extra" << std::endl;
    std::cerr.write(reinterpret_cast<const char*>(data), len);
    std::cerr << std::endl;
}

void ByBitRest::heartbeat_runner(void)
{
    while (heartbeat_thread_running) {
        std::this_thread::sleep_for(1s);
        if (DateTime::now() > heartbeat_timeout) {
            std::string uri = "https://api.bybit.com/public/linear/recent-trading-records?symbol=BTCUSDT&limit=1";
            request("GET", uri);
        }
    }
}

void ByBitRest::heartbeat_reset(void)
{
    heartbeat_timeout = DateTime::now() + 5s;
}
