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
        wallet(0.0f), entry_price(0.0f), pos_contracts(0.0f) {}

    void reset(void);

    void put_order(float price, float contracts);

private:
    sptrIntervals intervals;

    int intervals_idx_start;
    int intervals_idx_end;
    int intervals_idx;

    float wallet;
    float entry_price;
    float pos_contracts;

    void execute_order(float price, float contracts, bool taker);
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
