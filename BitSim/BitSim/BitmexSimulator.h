#pragma once
#include "pch.h"

#include "Intervals.h"
#include "RL_State.h"
#include "RL_Action.h"


class BitmexSimulator
{
public:
    BitmexSimulator(sptrIntervals intervals) :
        intervals(intervals),
        intervals_idx_start(0), intervals_idx_end(0),
        intervals_idx(0),
        wallet(0.0), pos_price(0.0), pos_contracts(0.0) {}

    void reset(void);
    double get_value(void);
    std::tuple<RL_State, bool> step(const RL_Action& action);

    void put_order(double price, double contracts);

private:
    sptrIntervals intervals;

    int intervals_idx_start;
    int intervals_idx_end;
    int intervals_idx;

    double wallet;
    double pos_price;
    double pos_contracts;

    void execute_order(double price, double contracts, bool taker);
    double liquidation_price(void);
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
