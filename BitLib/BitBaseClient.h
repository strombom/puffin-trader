#pragma once
#include "pch.h"

#include "Ticks.h"
#include "Intervals.h"


class BitBaseClient
{
public:
    BitBaseClient(void);

    sptrIntervals get_intervals(const std::string& symbol, 
                                const std::string& exchange,
                                const time_point_ms timestamp_start,
                                const time_point_ms timestamp_end,
                                std::chrono::milliseconds interval);

    sptrIntervals get_intervals(const std::string& symbol,
        const std::string& exchange,
        const time_point_ms timestamp_start,
        std::chrono::milliseconds interval);

    sptrTicks get_ticks(const std::string& symbol,
        const std::string& exchange,
        const time_point_ms timestamp_start,
        const time_point_ms timestamp_end);

    sptrTicks get_ticks(const std::string& symbol,
        const std::string& exchange,
        const time_point_ms timestamp_start);

private:
    zmq::context_t context;
    std::unique_ptr<zmq::socket_t> client;
};
