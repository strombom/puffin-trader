#include "pch.h"

#include "PolyFit.h"


PolyFit::PolyFit(int degree, int length) : degree(degree), length(length)
{
    for (auto i = 0; i < max_length; i++) {
        x[i] = i;
    }
}

void PolyFit::matrix_mul_a(void)
{

}

void PolyFit::matrix_mul_b(void)
{

}

void PolyFit::matrix_mul_c(void)
{

}

void PolyFit::matrix_t(void)
{
    for (auto i = 0, a = 0; i < PolyFitN; i++)
    {
        for (auto j = 0, b = 0; j < PolyFitN; j++)
        {
            KT[b + i] = x2[a + j];
            b += PolyFitN;
        }
        a += PolyFitN;
    }
}

void PolyFit::matrix_solve(void)
{
    matrix_t();
    //matrix_mul();
}

float PolyFit::calculate_direction(std::vector<double> y)
{
    for (auto i = 0, idx = 0; i < PolyFitN; i++) {
        y2[i] = 0;
        for (auto j = 0; j < PolyFitN; j++) {
            x2[idx + j] = 0;
        }
        idx += PolyFitN;
    }

    x2[0] = length;
    for (auto i = 0; i < PolyFitN; i++) {
        for (auto j = i + 1; j < PolyFitN; j++) {
            auto temp = 0.0;
            auto n = i + j;
            auto idx = 0;
            for (idx = 0; idx < length; idx++) {
                temp += std::pow(x[idx], n);
            }
            idx = j;
            for (n = i; n < PolyFitN; n++) {
                if (idx >= 0) {
                    x2[n * PolyFitN + idx] = temp;
                }
                idx--;
            }
        }
    }

    auto n = PolyFitN * 2 - 2;
    auto temp = 0.0;
    for (auto i = 0; i < length; i++) {
        temp += std::pow(x[i], n);
    }
    x2[PolyFitN * 2 - 1] = temp;
    for (auto i = 0; i < PolyFitN; i++) {
        temp = 0;
        for (auto j = 0; j < length; j++) {
            temp += y[j] * std::pow(x[j], i);
        }
        y2[i] = temp;
    }

    matrix_solve();

    return 1;
}
