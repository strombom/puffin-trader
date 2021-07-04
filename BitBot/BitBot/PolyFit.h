#pragma once

#include "pch.h"

#include "BitLib/BitBotConstants.h"

using namespace BitBot::Indicators;


class PolyFit
{
public:
    PolyFit(int degree, int length);

    float calculate_direction(std::vector<double> y);

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

    std::array<double, max_length> x;
    std::array<double, PolyFitN * PolyFitN> x2;
    std::array<double, PolyFitN> y2;
    std::array<double, PolyFitN> ks;

    std::array<double, PolyFitN * PolyFitN> KT;
    std::array<double, PolyFitN * PolyFitN> Kmul;
    std::array<double, PolyFitN * PolyFitN> Kinv;
    std::array<double, PolyFitN> Kb;

    std::array<double, PolyFitN * PolyFitN> L;
    std::array<double, PolyFitN * PolyFitN> U;

    std::array<double, PolyFitN> id;
    std::array<double, PolyFitN> ix;
    std::array<double, PolyFitN> ie;
};
