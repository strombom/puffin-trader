#include "pch.h"

#include "PolyFit.h"


PolyFit::PolyFit(int degree, int length) : degree(degree), length(length)
{
    for (auto i = 0; i < length; i++) {
        x[i] = i;
    }
}

bool PolyFit::matrix_lu(void)
{
    for (auto i = 0, a = 0; i < PolyFitN; i++) {
        for (auto j = 0; j < PolyFitN; j++) {
            L[a + j] = U[a + j] = 0;
        }
        U[a + i] = 1;
        a += PolyFitN;
    }
    for (auto j = 0, d = 0; j < PolyFitN; j++) {
        for (auto i = j, b = d; i < PolyFitN; i++) {
            auto temp = 0.0;
            auto a = 0, c = j;
            while (a < j) {
                temp += L[b + a] * U[c];
                c += PolyFitN;
                a++;
            }
            L[b + j] = Kmul[b + j] - temp;
            b += PolyFitN;
        }
        auto i = j + 1;
        while (i < PolyFitN) {
            auto temp = 0.0;
            auto a = 0, c = i;
            while (a < j) {
                temp += L[d + a] * U[c];
                a++;
                c += PolyFitN;
            }
            if (L[d + j] == 0) {
                return false;
            }
            U[d + i] = (Kmul[d + i] - temp) / L[d + j];
            i++;
        }
        d += PolyFitN;
    }
    return true;
}

bool PolyFit::matrix_inv(void)
{
    if (matrix_lu()) {
        auto a = PolyFitN * PolyFitN;
        for (auto i = 0; i < PolyFitN; i++) {
            ix[i] = id[i] = 0;
        }
        for (auto i = 0; i < PolyFitN; i++) {
            auto j = 0;
            for (j = 0; j < PolyFitN; j++) {
                ie[j] = 0;
            }
            ie[i] = 1;
            j = 0;
            auto b = 0;
            while (j < PolyFitN) {
                auto temp = 0.0;
                a = 0;
                while (a < j) {
                    temp += id[a] * L[b + a];
                    a++;
                }
                id[j] = ie[j] - temp;
                id[j] /= L[b + j];
                j++;
                b += PolyFitN;
            }
            j = PolyFitN - 1;
            b -= PolyFitN;
            while (j > -1) {
                auto temp = 0.0;
                a = j + 1;
                while (a < PolyFitN) {
                    temp += U[b + a] * ix[a];
                    a++;
                }
                ix[j] = id[j] - temp;
                ix[j] /= U[b + j];
                j--;
                b -= PolyFitN;
            }
            for (j = 0, b = i; j < PolyFitN; j++) {
                Kinv[b] = ix[j];
                b += PolyFitN;
            }
        }
    }
    else
    {
        return false;
    }

    return true;
}

void PolyFit::matrix_mul_a(void)
{
    for (auto i = 0; i < PolyFitN; i++) {
        for (auto j = 0; j < PolyFitN; j++) {
            const auto b = i * PolyFitN + j;
            Kmul[b] = 0;
            for (auto k = 0; k < PolyFitN; k++) {
                Kmul[b] += KT[i * PolyFitN + k] * x2[k * PolyFitN + j];
            }
        }
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

float PolyFit::calculate_direction(std::array<double, max_length> y)
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
            auto idx = 0;
            for (idx = 0; idx < length; idx++) {
                temp += std::pow(x[idx], i + j);
            }
            idx = j;
            for (auto n = i; n < PolyFitN; n++) {
                if (idx >= 0) {
                    x2[n * PolyFitN + idx] = temp;
                }
                idx--;
            }
        }
    }

    auto temp = 0.0;
    for (auto i = 0; i < length; i++) {
        temp += std::pow(x[i], PolyFitN * 2 - 2);
    }
    x2[PolyFitN * PolyFitN - 1] = temp;
    for (auto i = 0; i < PolyFitN; i++) {
        temp = 0;
        for (auto j = 0; j < length; j++) {
            temp += y[j] * std::pow(x[j], i);
        }
        y2[i] = temp;
    }

    if (matrix_solve()) {
        return 1.0;
    }

    return 0.0;
}
