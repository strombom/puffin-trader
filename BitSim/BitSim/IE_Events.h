#pragma once


class IE_Events
{
public:
    void append(std::chrono::milliseconds duration, float volume, float spread, int trade_count);
};

using sptrIE_Events = std::shared_ptr<IE_Events>;


class IE_Event
{
};

using sptrIE_Event = std::shared_ptr<IE_Event>;
