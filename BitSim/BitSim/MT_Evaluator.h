#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"


class MT_Evaluator
{
public:
    MT_Evaluator(sptrTicks ticks);

    void evaluate(void);

private:
    sptrTicks ticks;

};
