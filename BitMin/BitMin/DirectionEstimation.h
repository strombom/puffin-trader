#pragma once
#include "pch.h"

#include "DE_Server.h"

#include "BitLib/json11/json11.hpp"


class DirectionEstimation
{
public:
    DirectionEstimation(void);
    
    json11::Json get_directions(json11::Json parameters);
    

private:
    std::unique_ptr<DE_Server> de_server;
};

