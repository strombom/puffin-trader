#include "Server.h"
#include "Logger.h"

#include <zmq.hpp>
#include "json11.hpp"


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
        auto message_string = std::string(static_cast<char*>(message.data()), message.size());
        logger.info("Server::server_thread raw message (%s)", message_string.c_str());

        auto error_message = std::string{ "{\"command\":\"error\"}" };
        const auto command = json11::Json::parse(message_string.c_str(), error_message);
        const auto command_name = command["command"].string_value();
        
        if (command_name == "get_intervals") {
            logger.info("Server::server_thread get intervals!");

        }
        else {
            logger.info("Server::server_thread unknown command!");
        }

        auto send_result = server.send(message, zmq::send_flags::dontwait);
    }
}
