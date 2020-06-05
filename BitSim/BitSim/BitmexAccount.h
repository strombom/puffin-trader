#pragma once
#include "pch.h"



class BitmexAccount
{
public:
    BitmexAccount(void);

    double get_leverage(void);
    void limit_order(int contracts, double price);
    void market_order(int contracts);

private:

};

using sptrBitmexAccount = std::shared_ptr<BitmexAccount>;
