#pragma once
#include "pch.h"

#include "BinanceKlines.h"



class IntrinsicEvent
{
public:
    IntrinsicEvent(void) : timestamp(), price(0) {}
    IntrinsicEvent(time_point_ms timestamp, float price) : timestamp(timestamp), price(price) {}

    friend std::ostream& operator<<(std::ostream& stream, const IntrinsicEvent& row);
    friend std::istream& operator>>(std::istream& stream, IntrinsicEvent& row);

    time_point_ms timestamp;
    float price;
};


class IntrinsicEventRunner
{
public:
    std::vector<double> step(double price);

private:
    double current_price = 0.0;
    double previous_price = 0.0;
    double ie_start_price = 0.0;
    double ie_max_price = 0.0;
    double ie_min_price = 0.0;
    double ie_delta_top = 0.0;
    double ie_delta_bot = 0.0;
    bool initialized = false;
};


class IntrinsicEvents
{
public:
    void insert(BinanceKline binance_kline);
    void insert(sptrBinanceKlines binance_klines);

    void load(const std::string& file_path);
    void save(const std::string& file_path) const;

    IntrinsicEventRunner runner;
    std::vector<IntrinsicEvent> events;
};

