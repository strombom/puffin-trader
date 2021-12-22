#pragma once
#include "precompiled_headers.h"

#include "TickData.h"
#include "BitLib/DateTime.h"



class IntrinsicEvent
{
public:
    IntrinsicEvent(void) : timestamp(), price(0), size(0), tick_id(0) {}
    IntrinsicEvent(time_point_us timestamp, float price, float prev_low, float prev_high, float size, uint32_t tick_id) : timestamp(timestamp), price(price), size(size), tick_id(tick_id) {}

    //friend std::ostream& operator<<(std::ostream& stream, const IntrinsicEvent& row);
    friend std::istream& operator>>(std::istream& stream, IntrinsicEvent& row);

    time_point_us timestamp;
    float price;
    float size;
    uint32_t tick_id;
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
    IntrinsicEvents(void) {}
    IntrinsicEvents(const Symbol& symbol);

    void calculate_and_save(const Symbol& symbol, const TickData& tick_data);
    void join(void);

    void load(const Symbol& symbol);
    void save_csv(std::string file_path);
    double get_delta(void);

    std::vector<IntrinsicEvent> events;
    double delta;

private:
    std::vector<std::thread> threads;
};

using sptrIntrinsicEvents = std::shared_ptr<IntrinsicEvents>;
