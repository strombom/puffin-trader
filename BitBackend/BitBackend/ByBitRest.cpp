#include "ByBitRest.h"

#include <nghttp2/asio_http2_client.h>

//#include "url_parser.h"

using boost::asio::ip::tcp;


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

ByBitRest::ByBitRest(void)
{
    boost::system::error_code ec;
    boost::asio::io_service io_service;

    std::string uri = "https://api.bybit.com/public/linear/kline?symbol=BTCUSDT&interval=1&limit=2&from=1581231260";
    std::string scheme, host, service;

    if (nghttp2::asio_http2::host_service_from_uri(ec, scheme, host, service, uri)) {
        std::cerr << "error: bad URI: " << ec.message() << std::endl;
        return;
    }

    boost::asio::ssl::context tls_ctx(boost::asio::ssl::context::sslv23);
    tls_ctx.set_default_verify_paths();
    // disabled to make development easier...
    // tls_ctx.set_verify_mode(boost::asio::ssl::verify_peer);
    nghttp2::asio_http2::client::configure_tls_context(ec, tls_ctx);

    auto sess = scheme == "https" ? nghttp2::asio_http2::client::session(io_service, tls_ctx, host, service)
        : nghttp2::asio_http2::client::session(io_service, host, service);

    sess.on_connect([&sess, &uri](tcp::resolver::iterator endpoint_it) {
        std::cerr << "connected to " << (*endpoint_it).endpoint() << std::endl;
        boost::system::error_code ec;
        auto req = sess.submit(ec, "GET", uri, { {"cookie", {}} });
        if (ec) {
            std::cerr << "error: " << ec.message() << std::endl;
            return;
        }

        req->on_response([](const nghttp2::asio_http2::client::response& res) {
            std::cerr << "response header was received" << std::endl;
            print_header(res);

            res.on_data([](const uint8_t* data, std::size_t len) {
                std::cerr.write(reinterpret_cast<const char*>(data), len);
                std::cerr << std::endl;
                });
            });

        req->on_close([](uint32_t error_code) {
            std::cerr << "request done with error_code=" << error_code << std::endl;
            });

        req->on_push([](const nghttp2::asio_http2::client::request& push_req) {
            std::cerr << "push request was received" << std::endl;

            print_header(push_req);

            push_req.on_response([](const nghttp2::asio_http2::client::response& res) {
                std::cerr << "push response header was received" << std::endl;

                res.on_data([](const uint8_t* data, std::size_t len) {
                    std::cerr.write(reinterpret_cast<const char*>(data), len);
                    std::cerr << std::endl;
                    });
                });
            });
        });

    sess.on_error([](const boost::system::error_code& ec) {
        std::cerr << "error: " << ec.message() << std::endl;
        });

    io_service.run();
}
