#include "FE_Scheduler.h"


FE_Scheduler::FE_Scheduler(int n_iterations, double lr_max, double lr_min, double mo_max, double mo_min, int iteration, bool lr_test) :
    n_iterations(n_iterations), lr_max(lr_max), lr_min(lr_min), mo_max(mo_max), mo_min(mo_min), iteration(iteration), lr_test(lr_test)
{
    assert(iteration < n_iterations);
    if (lr_test) {
        log_file.open("C:\\development\\github\\puffin-trader\\tmp\\lr_test.txt");
    }

    lr_top_idx = (int)(n_iterations * (1 - annealing_part) / 2);
    annealing_idx = (int)(n_iterations * (1 - annealing_part));
}

std::tuple<double, double> FE_Scheduler::calc(double loss)
{
    ++iteration;

    if (lr_test) {
        return calc_lr_test(loss);
    }

    auto learning_rate = 0.0;
    auto momentum = 0.0;

    if (iteration < lr_top_idx) {
        // Learning rate rising
        const auto progress = (double)iteration / lr_top_idx;
        learning_rate = lr_min + progress * (lr_max - lr_min);
        momentum = mo_max - progress * (mo_max - mo_min);
    }
    else if (iteration < annealing_idx) {
        // Learning rate falling
        const auto progress = (double)(iteration - lr_top_idx) / (annealing_idx - lr_top_idx);
        learning_rate = lr_max - progress * (lr_max - lr_min);
        momentum = mo_max + progress * (mo_max - mo_min);
    }
    else {
        // Learning rate annealing
        const auto progress = (double)(iteration - annealing_idx) / (n_iterations - annealing_idx);
        learning_rate = lr_min - progress * (lr_min / 2);
        momentum = mo_max;
    }

    return std::make_tuple(learning_rate, momentum);
}

std::tuple<double, double> FE_Scheduler::calc_lr_test(double loss)
{
    static const auto lr_test_k = std::log(lr_max / lr_min) / (n_iterations - 1);
    const auto learning_rate = lr_min * std::exp(iteration * lr_test_k);
    log_file << iteration << "," << learning_rate << "," << loss << std::endl;
    return std::make_tuple(learning_rate, mo_max);
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
