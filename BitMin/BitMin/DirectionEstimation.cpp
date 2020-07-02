#include "pch.h"

#include "DirectionEstimation.h"


DirectionEstimation::DirectionEstimation(void)
{
    de_server = std::make_unique<DE_Server>();
}

json11::Json DirectionEstimation::get_directions(json11::Json parameters)
{
    return de_server->get_direction_data();
    //auto message_error = std::string{ "null" };
    //const auto message_string = std::string{ "abc" };
    //const auto message = json11::Json::parse(message_string.c_str(), message_error);
    //return message;
}
