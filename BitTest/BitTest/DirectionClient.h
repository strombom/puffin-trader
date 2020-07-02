#pragma once
#include "pch.h"

#include "BitLib/Intervals.h"


class DirectionClient
{
public:
    DirectionClient(void);

    void send(std::map<std::string, sptrIntervals> prices, std::vector<int> directions);

private:
    zmq::context_t context;
    std::unique_ptr<zmq::socket_t> client;
};

