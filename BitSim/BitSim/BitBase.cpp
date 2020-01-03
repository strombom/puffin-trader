
#include "BitBase.h"
#include "BitSimConstants.h"


BitBase::BitBase(void)
{
    client = std::make_unique<zmq::socket_t>(context, zmq::socket_type::req);
    client->connect(BitSim::BitBase::address);
}

void BitBase::get_intervals(void)
{
    auto message = zmq::message_t{ "abc" };
    client->send(message, zmq::send_flags::dontwait);

}
