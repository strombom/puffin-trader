#pragma once
#include "pch.h"

#include "Intervals.h"
#include "RL_State.h"
#include "RL_Action.h"


class BitmexSimulatorLogger
{
public:
    BitmexSimulatorLogger(const std::string &&filename);

    void log(double last_price, double order_size, double contracts, double wallet, double upnl);

private:
    std::ofstream file;

};

class BitmexSimulator
{
public:
    BitmexSimulator(sptrIntervals intervals, torch::Tensor features);

    RL_State reset(void);
    RL_State step(const RL_Action& action);
    std::tuple<double, double, double> calculate_position_leverage(double mark_price);

private:
    sptrIntervals intervals;
    torch::Tensor features;

    int intervals_idx_start;
    int intervals_idx_end;
    int intervals_idx;

    double start_value;
    double wallet;
    double pos_price;
    double pos_contracts;
    double previous_value;

    void market_order(double contracts);
    void limit_order(double contracts, double price);
    void execute_order(double contracts, double price, bool taker);
    bool is_liquidated(void);
    double get_reward(void);
    double liquidation_price(void);
    std::tuple<double, double, double> calculate_order_size(double buy_size, double sell_size);

    std::unique_ptr<BitmexSimulatorLogger> logger;
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
