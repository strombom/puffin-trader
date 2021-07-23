#pragma once
#include "pch.h"

#include "BinanceKlines.h"



class IntrinsicEvent
{
public:
    IntrinsicEvent(void) : timestamp(), price(0), delta(0) {}
    IntrinsicEvent(time_point_ms timestamp, float price, float delta) : timestamp(timestamp), price(price), delta(delta) {}

    friend std::ostream& operator<<(std::ostream& stream, const IntrinsicEvent& row);
    friend std::istream& operator>>(std::istream& stream, IntrinsicEvent& row);

    time_point_ms timestamp;
    float price;
    float delta;
};


class IntrinsicEventRunner
{
public:
    IntrinsicEventRunner(double delta) : delta(delta) {}
    std::vector<double> step(double price);
    double delta;

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
    IntrinsicEvents(std::string symbol);

    void calculate(sptrBinanceKlines binance_klines);

    void load(void);
    void save(void) const;

    std::string symbol;
    std::vector<IntrinsicEvent> events;
    double delta;
};

using sptrIntrinsicEvents = std::shared_ptr<IntrinsicEvents>;
