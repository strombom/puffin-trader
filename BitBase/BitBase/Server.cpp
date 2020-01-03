#include "Server.h"
#include "Logger.h"

#include <zmq.hpp>


Server::Server(sptrDatabase database) : 
    database(database), server_running(true)
{
    server_thread_handle = std::make_unique<std::thread>(&Server::server_thread, this);
}

Server::~Server(void)
{
    server_running = false;
    server_thread_handle->join();
}

void Server::server_thread(void)
{
    auto context = zmq::context_t{ 1 };
    auto server = zmq::socket_t{ context, zmq::socket_type::rep };
    server.bind("tcp://*:31000");
    server.setsockopt(ZMQ_RCVTIMEO, 500);

    while (server_running) {
        auto message = zmq::message_t{};
        auto recv_result = server.recv(message);
        if (!recv_result) {
            continue;
        }
        logger.info("Server::server_thread Message received");
        auto send_result = server.send(message, zmq::send_flags::dontwait);
    }
}
