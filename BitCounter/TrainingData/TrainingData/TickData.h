#pragma once
#include "precompiled_headers.h"

#include "BitLib/DateTime.h"
#include "Symbols.h"


class TickData
{
public:
    TickData(const Symbol& symbol);

    // Copy constructor
    TickData(const TickData& tick_data) : symbol(tick_data.symbol), timestamps(tick_data.timestamps), prices(tick_data.prices) {}

    void load(void);
    void load(time_point_us begin);
    void save(void) const;

    time_point_us get_timestamp_begin(void) const;
    time_point_us get_timestamp_end(void) const;

    friend std::ostream& operator<<(std::ostream& stream, const TickData& tick_data);
    friend std::istream& operator>>(std::istream& stream, TickData& tick_data);

    const std::string symbol;
    std::vector<time_point_us> timestamps;
    std::vector<float> prices;
};
