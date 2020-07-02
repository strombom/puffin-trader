#include "pch.h"

#include "BitLib/Logger.h"
#include "BitLib/DateTime.h"
#include "DE_Server.h"

#include <iostream>


DE_Server::DE_Server(void) :
    server_running(true)
{
    server_thread_handle = std::make_unique<std::thread>(&DE_Server::server_thread, this);
}

DE_Server::~DE_Server(void)
{
    server_running = false;
    server_thread_handle->join();
}

void DE_Server::server_thread(void)
{
    auto context = zmq::context_t{ 1 };
    auto server = zmq::socket_t{ context, zmq::socket_type::rep };
    server.bind("tcp://*:31005");
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

        std::cout << "Command" << std::endl;
        std::cout << command.dump() << std::endl;

        if (command_name == "set_directions") {
            direction_data = command;
            /*
            {"command": "set_directions", 
            "directions" : [2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0] , 
            "prices" : {"binance": [8618.9296875, 8618.4697265625, 8617.16015625, 8621.76953125, 8618.009765625, 8619.66015625, 8617.8203125, 8616.4296875, 8617.900390625, 8618.009765625, 8621.75, 8624.6201171875, 8623.990234375, 8624, 8624.8798828125, 8623.599609375, 8626.6904296875, 8630.8603515625, 8633.1298828125, 8634.51953125, 8634.900390625, 8634.23046875, 8637.3701171875, 8642.509765625, 8641.2802734375, 8640.099609375, 8640.75, 8643.48046875, 8640, 8641.5498046875] , "bitmex" : [8627.5, 8625.5, 8625.5, 8625.5, 8625.5, 8625.5, 8622.5, 8623.5, 8623, 8623, 8632, 8632.5, 8632.5, 8632.5, 8632.5, 8633, 8632.5, 8634.5, 8640, 8640, 8644.5, 8644.5, 8644, 8651, 8651, 8651, 8650.5, 8650.5, 8651, 8650.5] , "coinbase" : [8624.6796875, 8625.099609375, 8624.6796875, 8641.7998046875, 8641.7998046875, 8626.669921875, 8628.919921875, 8628.919921875, 8628.2802734375, 8627.6796875, 8630.1904296875, 8633.650390625, 8633.9501953125, 8633.9501953125, 8633.669921875, 8633.58984375, 8633.9599609375, 8644.099609375, 8644.099609375, 8645.5400390625, 8645.0400390625, 8648.08984375, 8650.009765625, 8651.099609375, 8652.2998046875, 8652.2998046875, 8652.2900390625, 8652.2900390625, 8652.2900390625, 8652.2998046875] }}
            */



            /*
            const auto intervals = database->get_intervals(command["exchange"].string_value(),
                command["symbol"].string_value(),
                DateTime::to_time_point_ms(command["timestamp_start"].string_value()),
                DateTime::to_time_point_ms(command["timestamp_end"].string_value()),
                std::chrono::seconds{ command["interval_seconds"].int_value() });

            auto buffer = std::stringstream{};
            buffer << *intervals;
            message = zmq::message_t{ buffer.str() };
            */
            message = zmq::message_t{ }; // Return empty message 
        }
        else {
            logger.info("DE_Server::server_thread unknown command!");
            message = zmq::message_t{ }; // Return empty message indicating error
        }

        const auto send_result = server.send(message, zmq::send_flags::dontwait);
    }
}

json11::Json DE_Server::get_direction_data(void)
{
    return direction_data;
}