#pragma once

#include <string>

class BitmexDaily
{
public:
    BitmexDaily(const std::string& _symbol);

private:
    std::string symbol;
};

