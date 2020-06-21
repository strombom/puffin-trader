#include "pch.h"

#include "BitLib/Logger.h"
#include "BitLib/DateTime.h"
#include "Server.h"


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
    server.bind("tcp://*:31001");
    server.setsockopt(ZMQ_RCVTIMEO, 500);

    auto message = zmq::message_t{};
    while (server_running) {
        const auto recv_result = server.recv(message);
        if (!recv_result) {
            continue;
        }
        const auto message_string = std::string(static_cast<char*>(message.data()), message.size());
        //logger.info("Server::server_thread raw message (%s)", message_string.c_str());

        auto error_message = std::string{ "{\"command\":\"error\"}" };
        const auto command = json11::Json::parse(message_string.c_str(), error_message);
        const auto command_name = command["command"].string_value();
        
        if (command_name == "get_intervals") {
            const auto intervals = database->get_intervals(command["exchange"].string_value(),
                                                           command["symbol"].string_value(),
                                                           DateTime::to_time_point_ms(command["timestamp_start"].string_value()),
                                                           DateTime::to_time_point_ms(command["timestamp_end"].string_value()),
                                                           std::chrono::seconds{ command["interval_seconds"].int_value() });

            auto buffer = std::stringstream{};
            buffer << *intervals;
            message = zmq::message_t{ buffer.str() };
        }
        else {
            logger.info("Server::server_thread unknown command!");
            message = zmq::message_t{ }; // Return empty message indicating error
        }

        const auto send_result = server.send(message, zmq::send_flags::dontwait);
    }
}
