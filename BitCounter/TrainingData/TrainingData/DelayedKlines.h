#pragma once
#include "precompiled_headers.h"

#include "TickData.h"
#include "IntrinsicEvents.h"
#include "BitLib/DateTime.h"


class DelayedKline
{
public:
    DelayedKline(void) : timestamp_open(), timestamp_close(), open(0), close(0), high(0), low(0), volume(0) {}
    DelayedKline(time_point_us timestamp_open, time_point_us timestamp_close)
        : timestamp_open(timestamp_open), timestamp_close(timestamp_close), open(0), close(0), high(0), low(0), volume(0) {}
    DelayedKline(time_point_us timestamp_open, time_point_us timestamp_close, float open, float close, float high, float low, float volume)
        : timestamp_open(timestamp_open), timestamp_close(timestamp_close), open(open), close(close), high(high), low(low), volume(volume) {}

    friend std::istream& operator>>(std::istream& stream, DelayedKline& row);

    time_point_us timestamp_open;
    time_point_us timestamp_close;
    float open;
    float close;
    float high;
    float low;
    float volume;
};


class DelayedKlines {
public:
    DelayedKlines(const IntrinsicEvents& intrinsic_events, const TickData& tick_data);

    void save_csv(std::string file_path);

private:
    std::vector<DelayedKline> klines;
};
