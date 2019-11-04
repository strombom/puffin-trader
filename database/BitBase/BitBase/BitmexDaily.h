#pragma once

#include "boost/thread.hpp"

#include "BitmexConstants.h"


enum class BitmexDailyState {
    Idle,
    Downloading
};

class BitmexDaily
{
public:
    BitmexDailyState get_state(void);
    void start_download(void);

private:
    BitmexDailyState state = BitmexDailyState::Idle;

    boost::thread* download_thread;
    void download(void);
};

