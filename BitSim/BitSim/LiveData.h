#pragma once

#include "BitBaseClient.h"

#include <thread>


class LiveData
{
public:
    LiveData(void);

private:
    BitBaseClient bitbase_client;
};
