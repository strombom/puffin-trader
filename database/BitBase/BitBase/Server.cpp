#include "Server.h"

#include <zmq.hpp>

Server::Server(void)
{
    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::push);
    sock.bind("inproc://test");
    const std::string_view m = "Hello, world";
    sock.send(zmq::buffer(m), zmq::send_flags::dontwait);

    /*
    zmq::context_t ctx;
    zmq::socket_t sock(ctx, zmq::socket_type::push);
    sock.bind("inproc://test");
    auto m = zmq::const_buffer("Hello", 5);
    sock.send(m, zmq::send_flags::dontwait);
    */
}
