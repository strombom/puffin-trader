#pragma once

#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)


class FE_Scheduler
{
public:
    FE_Scheduler(int n_iterations, double lr_max, double lr_min, double mo_max, double mo_min, int iteration = 0, bool lr_test = false) : 
        n_iterations(n_iterations), lr_max(lr_max), lr_min(lr_min), mo_max(mo_max), mo_min(mo_min), iteration(iteration), lr_test(lr_test)
    {
        assert(iteration < n_iterations);
    }
    
    std::tuple<double, double> calc(void);
    bool finished(void);

private:
    int iteration;
    int n_iterations;
    double lr_max;
    double lr_min;
    double mo_max;
    double mo_min;

    const bool lr_test;
};
