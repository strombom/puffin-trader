#pragma once
#include "pch.h"

#include "HttpRouter.h"

#include <chrono>
#include <thread>
#include <atomic>


class HttpListener;

class HttpServer : public std::enable_shared_from_this<HttpServer>
{
public:
    HttpServer(std::shared_ptr<HttpRouter> http_router);

    void start(void);
    void shutdown(void);

    template<class Body, class Allocator, class Send>
    void handle_request(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>>&& req, Send&& send);

    boost::asio::io_context ioc;
    boost::asio::ssl::context ctx;

private:
    std::shared_ptr<HttpRouter> http_router;

    std::atomic_bool server_thread_running;
    std::unique_ptr<std::thread> server_thread;

    std::shared_ptr<HttpListener> http_listener;

    boost::beast::string_view mime_type(boost::beast::string_view path);
    std::string path_cat(boost::beast::string_view base, boost::beast::string_view path);

    void load_server_certificate(boost::asio::ssl::context& ctx);
    void server_worker(void);
};

class HttpListener : public std::enable_shared_from_this<HttpListener>
{
public:
    HttpListener(std::shared_ptr<HttpServer> http_server, boost::asio::io_context& ioc, boost::asio::ssl::context& ctx, boost::asio::ip::tcp::endpoint endpoint);

    void run(void);

private:
    std::shared_ptr<HttpServer> http_server;
    boost::asio::io_context& ioc;
    boost::asio::ssl::context& ctx;
    boost::asio::ip::tcp::acceptor acceptor;

    void do_accept(void);
    void on_accept(boost::beast::error_code ec, boost::asio::ip::tcp::socket socket);
};

class HttpSession : public std::enable_shared_from_this<HttpSession>
{
public:
    explicit HttpSession(std::shared_ptr<HttpServer> http_server, boost::asio::ip::tcp::socket&& socket, boost::asio::ssl::context &ctx);

    void run(void);

    void on_run(void);
    void on_handshake(boost::beast::error_code ec);
    void do_read(void);
    void on_read(boost::beast::error_code ec, std::size_t bytes_transferred);
    void on_write(bool close, boost::beast::error_code ec, std::size_t bytes_transferred);
    void do_close(void);
    void on_shutdown(boost::beast::error_code ec);

private:
    struct send_lambda
    {
        HttpSession& self;

        explicit send_lambda(HttpSession& self) : self(self) {}

        template<bool isRequest, class Body, class Fields>
        void operator()(boost::beast::http::message<isRequest, Body, Fields>&& msg) const
        {
            // The lifetime of the message has to extend for the duration of the async operation so we use a shared_ptr to manage it.
            auto sp = std::make_shared<boost::beast::http::message<isRequest, Body, Fields>>(std::move(msg));

            // Store a type-erased version of the shared pointer in the class to keep it alive.
            self.res = sp;

            // Write the response
            boost::beast::http::async_write(self.stream, *sp, boost::beast::bind_front_handler(&HttpSession::on_write, self.shared_from_this(), sp->need_eof()));
        }
    };

    std::shared_ptr<HttpServer> http_server;

    boost::beast::ssl_stream<boost::beast::tcp_stream> stream;
    boost::beast::flat_buffer buffer;
    boost::beast::http::request<boost::beast::http::string_body> req;
    std::shared_ptr<void> res;
    send_lambda lambda;
};
