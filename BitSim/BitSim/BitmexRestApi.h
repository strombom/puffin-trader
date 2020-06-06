#pragma once
#include "pch.h"


class BitmexRestApi
{
public:

    const std::string limit_order(int contracts);

private:
    const std::string http_request(const std::string& host, const std::string& port, const std::string& target);

};
