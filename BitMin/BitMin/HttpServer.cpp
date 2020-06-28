#include "pch.h"

#include "HttpServer.h"
#include "BitMinConstants.h"

#include <iostream>


HttpServer::HttpServer(void) :
    ioc(),
    ctx(boost::asio::ssl::context::tlsv12),
    server_thread_running(true)
{
    load_server_certificate(ctx);
}

void HttpServer::start(void)
{
    auto address = boost::asio::ip::make_address(BitMin::HttpServer::address);
    auto endpoint = boost::asio::ip::tcp::endpoint{ address, static_cast<unsigned short>(BitMin::HttpServer::port) };
    http_listener = std::make_shared<HttpListener>(this->shared_from_this(), ioc, ctx, endpoint);
    http_listener->run();

    server_thread = std::make_unique<std::thread>(&HttpServer::server_worker, this);    
}

void HttpServer::shutdown(void)
{
    server_thread_running = false;

    try {
        server_thread->join();
    }
    catch (...) {}
}

void HttpServer::server_worker(void)
{
    //while (server_thread_running) {
        std::this_thread::sleep_for(500ms);
        std::cout << "start ioc" << std::endl;

        //const auto threads = 4;
        //auto io2 = ioc.get();
        /*
        std::vector<std::thread> v;
        v.reserve(threads - 1);
        for (auto i = threads - 1; i > 0; --i)
            v.emplace_back(
                [&ioc]
                {
                    ioc.run();
                });
        */
        ioc.run();
        //ioc->run();
        std::cout << "end ioc" << std::endl;
    //}
}

// This function produces an HTTP response for the given request. The type of the response object depends on the
//  contents of the request, so the interface requires the caller to pass a generic lambda for receiving the response.
template<class Body, class Allocator, class Send>
void HttpServer::handle_request(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>&& req, Send&& send)
{
    std::cout << "HttpServer::handle_request" << std::endl;

    // Returns a bad request response
    auto const bad_request = [&req](boost::beast::string_view why)
    {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::bad_request, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = std::string(why);
        res.prepare_payload();
        return res;
    };

    // Returns a not found response
    auto const not_found = [&req](boost::beast::string_view target)
    {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::not_found, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "The resource '" + std::string(target) + "' was not found.";
        res.prepare_payload();
        return res;
    };

    // Returns a server error response
    auto const server_error = [&req](boost::beast::string_view what)
    {
        boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::internal_server_error, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "An error occurred: '" + std::string(what) + "'";
        res.prepare_payload();
        return res;
    };

    // Make sure we can handle the method
    if (req.method() != boost::beast::http::verb::get && req.method() != boost::beast::http::verb::head) {
        return send(bad_request("Unknown HTTP-method"));
    }

    // Request path must be absolute and not contain "..".
    if (req.target().empty() || req.target()[0] != '/' || req.target().find("..") != boost::beast::string_view::npos) {
        return send(bad_request("Illegal request-target"));
    }

    // Build the path to the requested file
    std::string path = path_cat(BitMin::HttpServer::static_path, req.target());
    if (req.target().back() == '/') {
        path.append("index.html");
    }

    //path = boost::replace_all_copy(path, "/", std::string{ '\\' });

    // Attempt to open the file
    boost::beast::error_code ec;
    boost::beast::http::file_body::value_type body;
    body.open(path.c_str(), boost::beast::file_mode::scan, ec);

    // Handle the case where the file doesn't exist
    if (ec == boost::beast::errc::no_such_file_or_directory) {
        return send(not_found(req.target()));
    }

    // Handle an unknown error
    if (ec) {
        return send(server_error(ec.message()));
    }

    // Cache the size since we need it after the move
    auto const size = body.size();

    // Respond to HEAD request
    if (req.method() == boost::beast::http::verb::head) {
        boost::beast::http::response<boost::beast::http::empty_body> res{ boost::beast::http::status::ok, req.version() };
        res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(boost::beast::http::field::content_type, mime_type(path));
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }

    // Respond to GET request
    boost::beast::http::response<boost::beast::http::file_body> res{std::piecewise_construct, std::make_tuple(std::move(body)), std::make_tuple(boost::beast::http::status::ok, req.version()) };
    res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(boost::beast::http::field::content_type, mime_type(path));
    res.content_length(size);
    res.keep_alive(req.keep_alive());
    return send(std::move(res));
}

// Return a reasonable mime type based on the extension of a file.
boost::beast::string_view HttpServer::mime_type(boost::beast::string_view path)
{
    using boost::beast::iequals;
    auto const ext = [&path]
    {
        auto const pos = path.rfind(".");
        if (pos == boost::beast::string_view::npos)
            return boost::beast::string_view{};
        return path.substr(pos);
    }();
    if (iequals(ext, ".htm"))  return "text/html";
    if (iequals(ext, ".html")) return "text/html";
    if (iequals(ext, ".php"))  return "text/html";
    if (iequals(ext, ".css"))  return "text/css";
    if (iequals(ext, ".txt"))  return "text/plain";
    if (iequals(ext, ".js"))   return "application/javascript";
    if (iequals(ext, ".json")) return "application/json";
    if (iequals(ext, ".xml"))  return "application/xml";
    if (iequals(ext, ".swf"))  return "application/x-shockwave-flash";
    if (iequals(ext, ".flv"))  return "video/x-flv";
    if (iequals(ext, ".png"))  return "image/png";
    if (iequals(ext, ".jpe"))  return "image/jpeg";
    if (iequals(ext, ".jpeg")) return "image/jpeg";
    if (iequals(ext, ".jpg"))  return "image/jpeg";
    if (iequals(ext, ".gif"))  return "image/gif";
    if (iequals(ext, ".bmp"))  return "image/bmp";
    if (iequals(ext, ".ico"))  return "image/vnd.microsoft.icon";
    if (iequals(ext, ".tiff")) return "image/tiff";
    if (iequals(ext, ".tif"))  return "image/tiff";
    if (iequals(ext, ".svg"))  return "image/svg+xml";
    if (iequals(ext, ".svgz")) return "image/svg+xml";
    return "application/text";
}

// Append an HTTP rel-path to a local filesystem path.
// The returned path is normalized for the platform.
std::string HttpServer::path_cat(boost::beast::string_view base, boost::beast::string_view path)
{
    if (base.empty())
        return std::string(path);
    std::string result(base);
#ifdef BOOST_MSVC
    char constexpr path_separator = '\\';
    if (result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
    for (auto& c : result)
        if (c == '/')
            c = path_separator;
#else
    char constexpr path_separator = '/';
    if (result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
#endif
    return result;
}

void HttpServer::load_server_certificate(boost::asio::ssl::context& ctx)
{
    std::string const cert =
        "-----BEGIN CERTIFICATE-----\n"
        "MIIC8DCCAdigAwIBAgIUAth8oHSiJS92KOhhn9tor2PbLg4wDQYJKoZIhvcNAQEL\n"
        "BQAwFDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTIwMDYyODA5MDIxNloXDTIwMDcy\n"
        "ODA5MDIxNlowFDESMBAGA1UEAwwJbG9jYWxob3N0MIIBIjANBgkqhkiG9w0BAQEF\n"
        "AAOCAQ8AMIIBCgKCAQEA0OXwpvEAfOc16NZ2ul/jrZTtxaanF63mewVsipoleaBI\n"
        "OBBD+/5KSHwSNeztMpBkU339k2g3PC1Wc+85vbkXRBD5UiZToyzyZlK1E790VTYf\n"
        "vySCPKdlH8+w86XWUazzJH8roXYa//qYC6WfdigpOYtXWXkt4GUT6QdTDB4TfgVi\n"
        "BPFpFigJZLUWN214HrAvQHMMKY5+EnNUpynDrEHxtFZj8q2saNGMlyHrVWNvytFg\n"
        "xYssvkrlj6Kb4+QfbzZwMcrUhmUIMNGrmc56woUY4haJnw0ZN66eFd3jfhx/xazG\n"
        "t48KIiCNDxRdNRDCHMcAWySJOXOSV/PtYYhcGnOyEQIDAQABozowODAUBgNVHREE\n"
        "DTALgglsb2NhbGhvc3QwCwYDVR0PBAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMB\n"
        "MA0GCSqGSIb3DQEBCwUAA4IBAQCJaywRDeBoVeVapBcm5y21EuTRpsk/k1wnIejI\n"
        "21VIKiyNyc7ELhlShm8M8MsmG1Ydt/av6cxpB+vIoJ/2c0zUpPZxLLbrRD7SeNp7\n"
        "+kdkZl9VVJglpWjPhVR4lvsbm6VEOx0Rtk/3DUqy99RsUB2LLuc0X6vtAyToJ72d\n"
        "2CHiHjqzpXIEUJWqSpJwDg81aziPpgx/Q9/1F9xo2F9q0VpYKWj6KAuHyHQX7S5S\n"
        "qK1ZXZvz5cSaUcm/QjyPSPmsycRyEmplX7CTzxZ4d0crnIf0OB0nghBhTW1bTioK\n"
        "N+YpoewWArC572Pc+xebiKJOE3qEROj93cPoQIXhlCemMkfT\n"
        "-----END CERTIFICATE-----\n";

    std::string const key =
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDQ5fCm8QB85zXo\n"
        "1na6X+OtlO3FpqcXreZ7BWyKmiV5oEg4EEP7/kpIfBI17O0ykGRTff2TaDc8LVZz\n"
        "7zm9uRdEEPlSJlOjLPJmUrUTv3RVNh+/JII8p2Ufz7DzpdZRrPMkfyuhdhr/+pgL\n"
        "pZ92KCk5i1dZeS3gZRPpB1MMHhN+BWIE8WkWKAlktRY3bXgesC9Acwwpjn4Sc1Sn\n"
        "KcOsQfG0VmPyraxo0YyXIetVY2/K0WDFiyy+SuWPopvj5B9vNnAxytSGZQgw0auZ\n"
        "znrChRjiFomfDRk3rp4V3eN+HH/FrMa3jwoiII0PFF01EMIcxwBbJIk5c5JX8+1h\n"
        "iFwac7IRAgMBAAECggEBAIShmZvE5HRpMFc244cNnrb1GcomN2zXYsC0/uc5W5XP\n"
        "aEsAOMnj+DHxZHaMiUVTW/+OvNM8leVFO+C8AMMuo1D6HfJEl1HBFd5VoNZ69ler\n"
        "15g6i94Kwi/iZA0uO9b7ohyICaGoJRhleOfcLlSMxy/cAoeyYqhYRL3tgpLs/Z9C\n"
        "kgP71UZX5wwGOF9cVw8Xq+FVBfUaWcNIeS2tcZog6SD4NzYSnnMO2tttmUkSmLA3\n"
        "Y4mtiTXyE6aP1lwpGHqtMf8cOkSFpm4lWPelva71SIa40Xg2QwmrfhJl/E1zXeln\n"
        "uYgJL4ZTvRkIxFh33eR60UpNV/8XDXqV1W7AFgchJukCgYEA564F/doRh0DsuWef\n"
        "EmQDQfSNXhqb9+LZtEF9flvs61P37ohcWJ6kv8rsKPoWwHe2VeVtboz/WHOF3ZVI\n"
        "SF8TcLyX/eHKPVDWYyTQ0Mpf3oV5JlFsRpsy7s8X8HUiLQL0IHY/49hWfPyMGMpr\n"
        "wu/jMSoUaU9DW7xsGs/QGrB2BicCgYEA5tOz7dkwAezF0CJhAXYfD0amWhDEPaCO\n"
        "XWjFGAXPvp2wAEKZWyM9rr9vVCOVndKCJTacot3vV45BD7DKpLN/EfAJDi6UAbfo\n"
        "CzwdMVqgZstMx0XacCYxyLYFHhfCAijtOMgHgHXYRBeQ1c09tWgFphZCZmTGIXU6\n"
        "6vdGEBOtoQcCgYEAwS+jX2IPa7D/tDprsnIjJUQ+/VFC7RkEYAKcGlmc3T5PZ1/d\n"
        "YSgoSKV091ZS0nsAV+/PqHd6FM9/uJ1agL6G/2LDerhkY2yc2F9CoFegSXUS6JjM\n"
        "qLORbQDS/Xcu4EmuN8Ni0yL6O457Y9Cmuw49bCmii43EpSIhItpWtWqff9ECgYAo\n"
        "A7EgXVMuMj6Ee0E3wKNXwZaa7Gcoi8nCNaRdKry9vZZCJUlyfCR4Q6Vn/5czOcfn\n"
        "7ZSvwyt3kSiEMNCgRqGTmXnoZHix55CpSe2BncVKrJmNvfVFFcnxkmRGuDWgrDmb\n"
        "a0mpokQwYalw9thotzSrexZdq8CREkJ5cw5JQTALRwKBgQDNfe6Vxk3PnNOxkE/F\n"
        "wU+e6YRH4xT/fkTeCCaI3gi50Wjxkf7pljSJoZpdw1z+HHZG0g4SCEmqCmrrmAZG\n"
        "RKBrS7Awri6elygQqLbNyOyki2naJwQfR/c11MJYqW4wFpn9cyHcxvC8eG8kuj3A\n"
        "zkESeRDCoKcxjEXobJ+zrfc1hA==\n"
        "-----END PRIVATE KEY-----\n";

    std::string const dh =
        "-----BEGIN DH PARAMETERS-----\n"
        "MIICCAKCAgEAg3W3Q7KwdDkKjwIRTMiuCRiFIoKUKJI3YFPtgTzC57idSxJFkUrM\n"
        "iIHoFM5Abo0REBAuZbfPkSKPMCiScT8ljOEte003J5899UMnx/Mn2tV0vpb/HWyr\n"
        "EhV4KSmf2eFx0iN2McgjtzpvL9tExIy5UWRJaeGI+l1zFjOBDdw234JghQqZ8P52\n"
        "YzQmJQt17GchaM9s6QMwvqAIm9SfDwtJUexa0U4fh/qLnJnjpx882+CxGi8+e3JZ\n"
        "lWCU1yrDp7LqKdHu04bftqL518O9wyFlNdDiyH1ITxXlgVJHFpA/+HlHhfFEudHL\n"
        "ekLpvOlVJmkQDeWo+QbK8DDuLQE7Gz9aNW/D3ZvlhcONbqvddJqTZpkfxmnVV/x2\n"
        "dQbMHMYCKWochTcVrFfPflVN6d2FBu1eCpdbvCdzCe1I3z+vDb1LHRm3JZOKKFie\n"
        "NfT8t+vxvIgdQYClRw2uYylRfsOl2C838QnNXENHPCyw56PkyqANvxqgb3DauPZr\n"
        "GsQfjeBqFpeEtIOsIB52sVSHBl0FtbyfEJEcAguI34HNYiB6WAASDQPjIBwV9iEW\n"
        "Z7ZJKpOljMsrCElUFXq0//UjWH4266wCJHYGb1Ipcf9VSYR6XgD5B7dZ91q69OAL\n"
        "NaIWBRxQNQoobrIxXobYhN4eEGkOuBbubkCgfBmdnV8aX+faLHK1xgMCAQI=\n"
        "-----END DH PARAMETERS-----";

    ctx.set_password_callback(
        [](std::size_t,
            boost::asio::ssl::context_base::password_purpose)
        {
            return "test";
        });

    ctx.set_options(
        boost::asio::ssl::context::default_workarounds |
        boost::asio::ssl::context::no_sslv2 |
        boost::asio::ssl::context::single_dh_use);

    ctx.use_certificate_chain(
        boost::asio::buffer(cert.data(), cert.size()));

    ctx.use_private_key(
        boost::asio::buffer(key.data(), key.size()),
        boost::asio::ssl::context::file_format::pem);

    ctx.use_tmp_dh(
        boost::asio::buffer(dh.data(), dh.size()));
}

void fail(boost::beast::error_code ec, std::string message)
{
    std::string err = ec.message();
    if (ec.category() == boost::asio::error::get_ssl_category()) {
        err = std::string(" (")
            + boost::lexical_cast<std::string>(ERR_GET_LIB(ec.value())) + ","
            + boost::lexical_cast<std::string>(ERR_GET_FUNC(ec.value())) + ","
            + boost::lexical_cast<std::string>(ERR_GET_REASON(ec.value())) + ") "
            ;
        //ERR_PACK /* crypto/err/err.h */
        char buf[128];
        ::ERR_error_string_n(ec.value(), buf, sizeof(buf));
        err += buf;
    }

    std::cout << "Fail: " << ec.message() << " - " << ec.value() << " - " << message << std::endl;
    std::cout << "Fail2: " << err << std::endl;
}

HttpListener::HttpListener(std::shared_ptr<HttpServer> http_server, boost::asio::io_context& ioc, boost::asio::ssl::context& ctx, boost::asio::ip::tcp::endpoint endpoint) :
    http_server(http_server), ioc(ioc), ctx(ctx), acceptor(ioc)
{
    auto ec = boost::beast::error_code{};

    // Open the acceptor
    acceptor.open(endpoint.protocol(), ec);
    if (ec)
    {
        fail(ec, "open");
        return;
    }

    // Allow address reuse
    acceptor.set_option(boost::asio::socket_base::reuse_address(true), ec);
    if (ec)
    {
        fail(ec, "set_option");
        return;
    }

    // Bind to the server address
    acceptor.bind(endpoint, ec);
    if (ec)
    {
        fail(ec, "bind");
        return;
    }

    // Start listening for connections
    acceptor.listen(boost::asio::socket_base::max_listen_connections, ec);
    if (ec)
    {
        fail(ec, "listen");
        return;
    }
}

void HttpListener::run(void)
{
    do_accept();
}

void HttpListener::do_accept(void)
{
    std::cout << "HttpListener::do_accept" << std::endl;

    // The new connection gets its own strand
    acceptor.async_accept(boost::asio::make_strand(ioc), boost::beast::bind_front_handler(&HttpListener::on_accept, shared_from_this()));
}

void HttpListener::on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket)
{
    std::cout << "HttpListener::on_accept" << std::endl;

    if (ec) {
        fail(ec, "accept");
    }
    else {
        // Create the session and run it
        auto session = std::make_shared<HttpSession>(http_server, std::move(socket), ctx);
        session->run();
    }

    // Accept another connection
    do_accept();
}


HttpSession::HttpSession(std::shared_ptr<HttpServer> http_server, boost::asio::ip::tcp::socket&& socket, boost::asio::ssl::context& ctx) :
    http_server(http_server), stream(std::move(socket), ctx), lambda(*this)
{

}

void HttpSession::run(void)
{
    // We need to be executing within a strand to perform async operations on the I/O objects in this session. Although not strictly necessary
    // for single-threaded contexts, this example code is written to be thread-safe by default.
    boost::asio::dispatch(stream.get_executor(), boost::beast::bind_front_handler(&HttpSession::on_run, shared_from_this()));
}

void HttpSession::on_run(void)
{
    std::cout << "HttpSession::on_run" << std::endl;

    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(30s);

    // Perform the SSL handshake
    stream.async_handshake(boost::asio::ssl::stream_base::server, boost::beast::bind_front_handler(&HttpSession::on_handshake, shared_from_this()));
}

void HttpSession::on_handshake(boost::beast::error_code ec)
{
    std::cout << "HttpSession::on_handshake" << std::endl;

    if (ec) {
        return fail(ec, "handshake");
    }

    do_read();
}

void HttpSession::do_read(void)
{
    std::cout << "HttpSession::do_read" << std::endl;

    // Make the request empty before reading, otherwise the operation behavior is undefined.
    req = {};

    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(30s);

    // Read a request
    boost::beast::http::async_read(stream, buffer, req, boost::beast::bind_front_handler(&HttpSession::on_read, shared_from_this()));
}

void HttpSession::on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    std::cout << "HttpSession::on_read" << std::endl;

    boost::ignore_unused(bytes_transferred);

    // This means they closed the connection
    if (ec == boost::beast::http::error::end_of_stream)
        return do_close();

    if (ec)
        return fail(ec, "read");

    // Send the response
    http_server->handle_request(std::move(req), lambda);
}

void HttpSession::on_write(bool close, boost::beast::error_code ec, std::size_t bytes_transferred)
{
    std::cout << "HttpSession::on_write" << std::endl;

    boost::ignore_unused(bytes_transferred);

    if (ec)
        return fail(ec, "write");

    if (close)
    {
        // This means we should close the connection, usually because
        // the response indicated the "Connection: close" semantic.
        return do_close();
    }

    // We're done with the response so delete it
    res = nullptr;

    // Read another request
    do_read();
}

void HttpSession::do_close(void)
{
    std::cout << "HttpSession::do_close" << std::endl;

    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

    // Perform the SSL shutdown
    stream.async_shutdown(boost::beast::bind_front_handler(&HttpSession::on_shutdown, shared_from_this()));
}

void HttpSession::on_shutdown(boost::beast::error_code ec)
{
    std::cout << "HttpSession::on_shutdown" << std::endl;

    if (ec)
        return fail(ec, "shutdown");

    // At this point the connection is closed gracefully
}
