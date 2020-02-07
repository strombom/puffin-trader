#pragma once
#include "pch.h"


class FE_Scheduler
{
public:
    FE_Scheduler(int n_iterations, double lr_max, double lr_min, double mo_max, double mo_min, int iteration = 0, bool lr_test = false);

    std::tuple<double, double> calc(double loss);
    bool finished(void);

private:
    int iteration;
    int n_iterations;
    double lr_max;
    double lr_min;
    double mo_max;
    double mo_min;
    const bool lr_test;

    const double lr_top_part = 0.2;
    const double annealing_part = 0.1;
    int lr_top_idx;
    int annealing_idx;
    std::ofstream log_file;

    std::tuple<double, double> calc_lr_test(double loss);
};

using uptrFE_Scheduler = std::unique_ptr<FE_Scheduler>;
