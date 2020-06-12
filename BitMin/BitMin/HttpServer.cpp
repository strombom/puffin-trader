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
    acceptor.async_accept(
        boost::asio::make_strand(*ioc),
        boost::beast::bind_front_handler(
            &HttpListener::on_accept,
            shared_from_this()));
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
        /*
        std::make_shared<session>(
            std::move(socket),
            ctx_,
            doc_root_)->run();
        */
    }

    // Accept another connection
    do_accept();
}
