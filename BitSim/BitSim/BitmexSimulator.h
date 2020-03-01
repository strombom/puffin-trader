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
    RL_State step(const RL_Action& action);

private:
    sptrIntervals intervals;

    int intervals_idx_start;
    int intervals_idx_end;
    int intervals_idx;

    double wallet;
    double pos_price;
    double pos_contracts;

    void market_order(double contracts);
    void limit_order(double contracts, double price);
    void execute_order(double contracts, double price, bool taker);
    double liquidation_price(void);
    double sigmoid_to_price(double price, double sigmoid);
    std::tuple<double, double> calculate_order_size(double buy_size, double sell_size);
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
