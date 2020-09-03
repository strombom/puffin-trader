#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"


class MT_Direction {
public:
    MT_Direction(bool __up) : _up(__up) {}

    static const bool up = true;
    static const bool down = false;

    bool operator==(bool __up) {
        return __up == _up;
    }

private:
    bool _up;
};

class MT_OrderBookBuffer {
public:
    MT_OrderBookBuffer(void);

    void step(time_point_ms timestamp, double price);

    std::tuple<double, double> get_price(time_point_ms timestamp);

private:
    static constexpr int size = 1000;
    std::array<time_point_ms, size> timestamps;
    std::array<double, size> prices;
    int length;
    int next_idx;

    double order_book_bottom;
};

class MT_OrderBook {
public:
    MT_OrderBook(time_point_ms timestamp, double price);

    bool update(time_point_ms timestamp, double price, MT_Direction direction);

    MT_OrderBookBuffer buffer;
private:
};

class MT_Simulator
{
public:
    MT_Simulator(const Tick& first_tick);

    void step(const Tick& tick);

    void market_order(double contracts);
    void limit_order(double contracts, double price);

private:
    double wallet;
    double pos_price;
    double pos_contracts;
    double time_since_leverage_change;

    MT_OrderBook order_book;

    void execute_order(double contracts, double price, double fee);
    double calculate_order_size(double leverage);
    void market_order(double contracts, bool use_fee);
    void limit_order(double contracts, double price, bool use_fee);
    std::tuple<double, double, double> calculate_position_leverage(double mark_price);
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
