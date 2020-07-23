#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"


class MT_Simulator
{
public:
    MT_Simulator();

    void step(const Tick& tick);

    void market_order(double contracts);
    void limit_order(double contracts, double price);

private:
    double wallet;
    double pos_price;
    double pos_contracts;
    double time_since_leverage_change;
    double orderbook_last_price;

    void execute_order(double contracts, double price, double fee);
    double calculate_order_size(double leverage);
};

/*
class BitmexSimulator
{
public:
    BitmexSimulator(sptrIntervals intervals, torch::Tensor features);

    sptrRL_State reset(int idx_episode, bool validation, double training_progress);
    sptrRL_State step(sptrRL_Action action, bool last_step);
    std::tuple<double, double, double> calculate_position_leverage(double mark_price);

    time_point_ms get_start_timestamp(void);

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
    double time_since_leverage_change;

    double orderbook_last_price;
    double training_progress;

    double get_reward_previous_value;

    void market_order(double contracts);
    void market_order(double contracts, bool use_fee);
    void limit_order(double contracts, double price);
    void limit_order(double contracts, double price, bool use_fee);
    void execute_order(double contracts, double price, double fee);
    bool is_liquidated(void);
    double get_reward(void);
    double liquidation_price(void);
    double calculate_order_size(double leverage);

    std::unique_ptr<BitmexSimulatorLogger> logger;
};

using sptrBitmexSimulator = std::shared_ptr<BitmexSimulator>;
*/
