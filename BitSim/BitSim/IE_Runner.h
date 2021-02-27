#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "IE_Events.h"


class IE_Runner
{
public:
    IE_Runner(double delta, float initial_price, time_point_ms initial_timestamp);

    sptrIE_Event step(const Tick &tick);

private:

};

