#include "pch.h"

#include "PolyFit.h"


PolyFit::PolyFit(int degree, int length) : degree(degree), length(length)
{
    for (auto i = 0; i < max_length; i++) {
        x[i] = i;
    }
}

bool PolyFit::matrix_inv(void)
{
    return true;
}

void PolyFit::matrix_mul_a(void)
{
    for (auto i = 0, a = 0, c = 0; i < PolyFitN; i++) {
        for (auto j = 0; j < PolyFitN; j++)
        {
            const auto b = a + j;
            Kmul[b] = 0;
            for (auto k = 0, d = 0; k < PolyFitN * PolyFitN; k++)
            {
                Kmul[b] += KT[c + k] * x2[d + j];
                d += PolyFitN;
            }
        }
        c += PolyFitN * PolyFitN;
        a += PolyFitN;
    }
}

void PolyFit::matrix_mul_b(void)
{
    for (auto i = 0; i < PolyFitN; i++) {
        Kb[i] = 0;
        for (auto k = 0; k < PolyFitN; k++) {
            Kb[i] += KT[i * PolyFitN + k] * y2[k];
        }
    }
}

void PolyFit::matrix_mul_c(void)
{
    for (auto i = 0; i < PolyFitN; i++) {
        ks[i] = 0;
        for (auto k = 0; k < PolyFitN; k++) {
            ks[i] += Kinv[i * PolyFitN + k] * Kb[k];
        }
    }
}

void PolyFit::matrix_t(void)
{
    for (auto i = 0, a = 0; i < PolyFitN; i++) {
        for (auto j = 0, b = 0; j < PolyFitN; j++) {
            KT[b + i] = x2[a + j];
            b += PolyFitN;
        }
        a += PolyFitN;
    }
}

bool PolyFit::matrix_solve(void)
{
    matrix_t();
    matrix_mul_a();
    matrix_mul_b();
    if (matrix_inv()) {
        matrix_mul_c();
        return true;
    }
    return false;
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
