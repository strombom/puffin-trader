#pragma once

#include "Intervals.h"

#pragma warning(push)
#pragma warning(disable: 4005)
#include <zmq.hpp>
#pragma warning(pop)


class BitBaseClient
{
public:
    BitBaseClient(void);

    std::unique_ptr<Intervals> get_intervals(const std::string& symbol, 
                                             const std::string& exchange,
                                             const time_point_s timestamp_start,
                                             const time_point_s timestamp_end,
                                             std::chrono::seconds interval);

private:
    zmq::context_t context;
    std::unique_ptr<zmq::socket_t> client;

};
