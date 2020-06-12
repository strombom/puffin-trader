#include "pch.h"

#include "HttpServer.h"
#include "BitMinConstants.h"

#include <iostream>


HttpServer::HttpServer(void) :
    server_thread_running(true)
{

}

void HttpServer::start(void)
{
    auto endpoint = boost::asio::ip::tcp::endpoint{ boost::asio::ip::make_address(BitMin::HttpServer::address), static_cast<unsigned short>(BitMin::HttpServer::port) };
    auto http_listener = std::make_shared<HttpListener>(ioc, ctx, endpoint);
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
    while (server_thread_running) {
        std::this_thread::sleep_for(500ms);
    }
}

//HttpListener::HttpListener(std::shared_ptr<boost::asio::io_context> ioc, std::shared_ptr<boost::asio::ssl::context> ctx, boost::asio::ip::tcp::endpoint endpoint) //: // ioc(ioc), ctx(ctx), acceptor(ioc)

void fail(boost::beast::error_code ec, std::string message)
{
    std::cout << "Fail: " << ec.message() << " - " << message << std::endl;
}

HttpListener::HttpListener(std::shared_ptr<boost::asio::io_context> ioc, std::shared_ptr<boost::asio::ssl::context> ctx, boost::asio::ip::tcp::endpoint endpoint) :
    ioc(ioc), ctx(ctx), acceptor(*ioc)
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
    // The new connection gets its own strand
    acceptor.async_accept(boost::asio::make_strand(*ioc), boost::beast::bind_front_handler(&HttpListener::on_accept, shared_from_this()));
}

void HttpListener::on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket)
{
    if (ec)
    {
        fail(ec, "accept");
    }
    else
    {
        // Create the session and run it
        std::make_shared<HttpSession>(
            std::move(socket),
            ctx)->run();
    }

    // Accept another connection
    do_accept();
}


HttpSession::HttpSession(boost::asio::ip::tcp::socket&& socket, std::shared_ptr<boost::asio::ssl::context> ctx) :
    stream(std::move(socket), *ctx), lambda(*this)
{

}

void HttpSession::run(void)
{
    // We need to be executing within a strand to perform async operations
    // on the I/O objects in this session. Although not strictly necessary
    // for single-threaded contexts, this example code is written to be
    // thread-safe by default.
    boost::asio::dispatch(stream.get_executor(), boost::beast::bind_front_handler(&HttpSession::on_run, shared_from_this()));
}

void HttpSession::on_run(void)
{
    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(30s);

    // Perform the SSL handshake
    stream.async_handshake(boost::asio::ssl::stream_base::server, boost::beast::bind_front_handler(&HttpSession::on_handshake, shared_from_this()));
}

void HttpSession::on_handshake(boost::beast::error_code ec)
{
    if (ec)
        return fail(ec, "handshake");

    do_read();
}

void HttpSession::do_read(void)
{
    // Make the request empty before reading,
    // otherwise the operation behavior is undefined.
    req = {};

    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(30s);

    // Read a request
    boost::beast::http::async_read(stream, buffer, req, boost::beast::bind_front_handler(&HttpSession::on_read, shared_from_this()));
}

void HttpSession::on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    // This means they closed the connection
    if (ec == boost::beast::http::error::end_of_stream)
        return do_close();

    if (ec)
        return fail(ec, "read");

    // Send the response
    //handle_request(std::move(req), lambda);
}

void HttpSession::on_write(bool close, boost::beast::error_code ec, std::size_t bytes_transferred)
{
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
    // Set the timeout.
    boost::beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));

    // Perform the SSL shutdown
    stream.async_shutdown(boost::beast::bind_front_handler(&HttpSession::on_shutdown, shared_from_this()));
}

void HttpSession::on_shutdown(boost::beast::error_code ec)
{
    if (ec)
        return fail(ec, "shutdown");

    // At this point the connection is closed gracefully
}
