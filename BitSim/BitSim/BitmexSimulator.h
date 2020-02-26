#pragma once
#include "pch.h"

#include "Intervals.h"


class BitmexSimulator
{
public:
    BitmexSimulator(sptrIntervals intervals) :
        intervals(intervals),
        intervals_idx_start(0), intervals_idx_end(0),
        intervals_idx(0),
        wallet(0.0), entry_price(0.0), pos_contracts(0.0) {}

    void reset(void);

    void put_order(double price, double contracts);

private:
    sptrIntervals intervals;

    int intervals_idx_start;
    int intervals_idx_end;
    int intervals_idx;

    double wallet;
    double entry_price;
    double pos_contracts;

    void execute_order(double price, double contracts, bool taker);
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
