#include "pch.h"
#include "MT_Simulator.h"
#include "MT_Evaluator.h"
#include "PD_Events.h"


MT_Evaluator::MT_Evaluator(sptrTicks ticks) :
    ticks(ticks)
{
    std::cout << ticks->rows[0].price << std::endl;
    evaluate();
}

void MT_Evaluator::evaluate(void)
{
    const auto first_tick = ticks->rows[0];
    auto simulator = MT_Simulator{ first_tick };
    auto events = PD_Events{ first_tick };

    for (auto row_idx = 0; row_idx < ticks->rows.size(); ++row_idx) {
        const auto &tick = ticks->rows[row_idx];

        auto event = events.step(tick);
        simulator.step(tick);

        if (event != nullptr) {
            std::cout << event->price << std::endl;



        }
    }
}

/*
    event

    buy/nobuy
    stoploss
    minprofit
*/

