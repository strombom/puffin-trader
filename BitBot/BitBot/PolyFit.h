#pragma once

#include "pch.h"

#include "BitLib/BitBotConstants.h"

using namespace BitBot::Indicators;


class PolyFit
{
public:
    PolyFit(void);

    std::tuple<double, double> calculate_direction(std::array<double, max_length> price_steps, int degree, int length);

private:
    std::array<double, max_degree_p1> coeffs;
    std::array<double, max_degree_t2p1> X;
    std::array<double, max_degree_p1> Y;
    std::array<std::array<double, max_degree_p2>, max_degree_p1> B;
};
