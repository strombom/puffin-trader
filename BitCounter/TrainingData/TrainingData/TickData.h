#pragma once
#include "precompiled_headers.h"

#include "BitLib/DateTime.h"
#include "Symbols.h"


class Tick
{
public:
    Tick(time_point_us timestamp, float price, float size)
        : timestamp(timestamp), price(price), size(size) {}

    time_point_us timestamp;
    float price;
    float size;
};

class TickData
{
public:
    TickData(const Symbol& symbol);

    void save_csv(std::string file_path);
    std::vector<Tick> rows;
    //std::vector<time_point_us> timestamps;
    //std::vector<float> prices;
};

using sptrTickData = std::shared_ptr<TickData>;
