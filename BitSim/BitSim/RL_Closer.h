#pragma once
#include "pch.h"

#include "FE_Observations.h"


class RL_Closer
{
public:
    RL_Closer(sptrFE_Observations observations) :
        observations(observations) {}

    void train(void);

private:
    sptrFE_Observations observations;
    
private:

};
