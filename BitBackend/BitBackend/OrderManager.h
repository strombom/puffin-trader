#pragma once
#include "precompiled_headers.h"

#include "Portfolio.h"


class OrderManager
{
public:
    OrderManager(sptrPortfolio portfolio) : 
        portfolio(portfolio) {}

private:
    sptrPortfolio portfolio;
};