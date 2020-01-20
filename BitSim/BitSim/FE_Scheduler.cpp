#include "FE_Scheduler.h"


std::tuple<double, double> FE_Scheduler::calc(void)
{
    ++iteration;

    if (lr_test) {
        static const auto lr_test_k = std::log(lr_max / lr_min) / (n_iterations - 1);
        const auto learning_rate = lr_min * std::exp(iteration * lr_test_k);
        
        return std::make_tuple(learning_rate, mo_max);
    }



    return std::make_tuple(1.0, mo_max);
}

bool FE_Scheduler::finished(void)
{
    if (iteration >= n_iterations) {
        return true;
    }
    else {
        return false;
    }
}
