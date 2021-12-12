#pragma once
#include "precompiled_headers.h"

#include "BitLib/DateTime.h"
#include "Symbols.h"


class TickData
{
public:
    TickData(const Symbol& symbol);

    void save_csv(std::string file_path);
    std::vector<std::tuple<time_point_us, float, float>> rows;
    //std::vector<time_point_us> timestamps;
    //std::vector<float> prices;
};
