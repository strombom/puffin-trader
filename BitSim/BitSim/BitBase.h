#pragma once

#pragma warning(push)
#pragma warning(disable: 4005)
#include <zmq.hpp>
#pragma warning(pop)


class BitBase
{
public:
    BitBase(void);

    void get_intervals(void);

private:
    zmq::context_t context;
    std::unique_ptr<zmq::socket_t> client;

};
