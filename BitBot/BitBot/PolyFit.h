#pragma once

#include "pch.h"

#include "BitLib/BitBotConstants.h"

using namespace BitBot::Indicators;


class PolyFit
{
public:
    PolyFit(void);

    std::tuple<double, double> calculate_direction(std::array<double, max_length> price_steps, int degree, int length);

    bool matrix_solve(void);
    void matrix_t(void);
    void matrix_mul_a(void);
    void matrix_mul_b(void);
    void matrix_mul_c(void);
    bool matrix_inv(void);
    bool matrix_lu(void);

private:
    int degree;
    int length;
    int polyfit_n;

    std::array<double, max_length> x;
    std::array<double, max_polyfit_n * max_polyfit_n> x2;
    std::array<double, max_polyfit_n> y2;
    std::array<double, max_polyfit_n> ks;

    std::array<double, max_polyfit_n * max_polyfit_n> KT;
    std::array<double, max_polyfit_n * max_polyfit_n> Kmul;
    std::array<double, max_polyfit_n * max_polyfit_n> Kinv;
    std::array<double, max_polyfit_n> Kb;

    std::array<double, max_polyfit_n * max_polyfit_n> L;
    std::array<double, max_polyfit_n * max_polyfit_n> U;

    std::array<double, max_polyfit_n> id;
    std::array<double, max_polyfit_n> ix;
    std::array<double, max_polyfit_n> ie;
};
