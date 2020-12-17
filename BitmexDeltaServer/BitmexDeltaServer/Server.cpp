
#include "Server.h"

#include "zmq.hpp"
#include <msgpack.hpp>
#include "json11.hpp"

#include <memory>
#include <iostream>


Server::Server(sptrTickData tick_data) :
    tick_data(tick_data), server_running(true)
{
    server_thread_handle = std::make_unique<std::thread>(&Server::server_thread, this);
}

Server::~Server(void)
{
    server_running = false;
    server_thread_handle->join();
}

void Server::test(void)
{
    test_condition.notify_one();
}

void Server::server_thread(void)
{
    auto context = zmq::context_t{ 1 };
    auto server = zmq::socket_t{ context, zmq::socket_type::rep };
    server.bind(Bitmex::server_address);
    server.setsockopt(ZMQ_RCVTIMEO, 500);
    server.setsockopt(ZMQ_SNDTIMEO, 2000);

    auto message = zmq::message_t{};
    while (server_running) {
        {
            //auto test_lock = std::unique_lock<std::mutex>{ test_mutex };
            //test_condition.wait(test_lock);
        }

        const auto recv_result = server.recv(message);
        if (!recv_result) {
            continue;
        }


        try {
            const auto message_string = std::string(static_cast<char*>(message.data()), message.size());

            std::cout << "Rcv: " << message_string << std::endl;

            //logger.info("Server::server_thread raw message (%s)", message_string.c_str());

            auto error_message = std::string{};
            const auto command = json11::Json::parse(message_string.c_str(), error_message);
            const auto command_name = command["command"].string_value();
            std::cout << "Rcv command: " << command_name << std::endl;


            if (command_name == "get_ticks") {
                //logger.info("Server::server_thread get intervals!");
                //const auto intervals = database->get_intervals(command["exchange"].string_value(),
                //    command["symbol"].string_value(),
                //    DateTime::to_time_point_ms(command["timestamp_start"].string_value()),
                //    DateTime::to_time_point_ms(command["timestamp_end"].string_value()),
                //    std::chrono::seconds{ command["interval_seconds"].int_value() });

                const auto symbol = command["symbol"].string_value();
                const auto timestamp = DateTime::to_time_point(command["timestamp_start"].string_value(), "%FT%TZ");
                const auto max_rows = command["max_rows"].int_value();

                auto ticks = tick_data->get(symbol, timestamp, max_rows);

                auto sbuf = msgpack::sbuffer{};
                msgpack::pack(sbuf, ticks);
                message = zmq::message_t{ sbuf.size() };
                memcpy(message.data(), sbuf.data(), sbuf.size());

                std::cout << "Send data, size: " << sbuf.size() << std::endl;
            }
            else {
                //logger.info("Server::server_thread unknown command!");
                message = zmq::message_t{};
            }

            server.send(message, zmq::send_flags::dontwait);

        }
        catch (std::exception& e) {
        
        }

    }
}
