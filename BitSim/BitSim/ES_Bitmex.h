#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"
#include "BitLib/Intervals.h"


struct ES_State
{
public:

};

class ES_Bitmex
{
public:
    ES_Bitmex(void);

    void reset(double price);

    ES_State market_order(double price, double volume);
    double get_leverage(double price);
    double calculate_order_size(double leverage, double mark_price);

private:
    double wallet;
    double pos_price;
    double pos_contracts;
    double start_value;
    double previous_value;

    std::tuple<double, double, double> calculate_position_leverage(double mark_price);
};

using sptrES_Bitmex = std::shared_ptr<ES_Bitmex>;


/*
class ES_BitmexLogger
{
public:
    ES_BitmexLogger(const std::string &filename, bool enabled);

    void log(
        double last_price,
        double wallet,
        double upnl,
        double position_contracts,
        double position_leverage,
        int make_order,
        double order_leverage,
        //double order_contracts,
        //double order_leverage,
        //int order_idle,
        //int order_limit,
        //int order_market,
        double reward
    );

private:
    std::ofstream file;
    bool enabled;
};

class ES_Bitmex
{
public:
    ES_Bitmex(void);
    
    void reset(void);

    void market_order(double contracts);
    void limit_order(double contracts, double price);


    //sptrRL_State step(sptrRL_Action action);
    std::tuple<double, double, double> calculate_position_leverage(double mark_price);

    time_point_ms get_start_timestamp(void);

private:
    //sptrIntervals intervals;
    //torch::Tensor features;

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

    void market_order(double contracts, bool use_fee);
    void limit_order(double contracts, double price, bool use_fee);
    void execute_order(double contracts, double price, double fee);
    bool is_liquidated(void);
    double get_reward(void);
    double liquidation_price(void);
    double calculate_order_size(double leverage);

    std::unique_ptr<ES_BitmexLogger> logger;
};

using sptrES_Bitmex = std::shared_ptr<ES_Bitmex>;

*/
