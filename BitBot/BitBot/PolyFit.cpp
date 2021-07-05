#include "pch.h"

#include "PolyFit.h"

// https://github.com/gullibility/leastsquare


PolyFit::PolyFit(void) : degree(degree), length(length)
{
    for (auto i = 0; i < max_length; i++) {
        x[i] = i;
    }
}

bool PolyFit::matrix_lu(void)
{
    for (auto i = 0; i < polyfit_n; i++) {
        for (auto j = 0; j < polyfit_n; j++) {
            L[i * polyfit_n + j] = U[i * polyfit_n + j] = 0;
        }
        U[i * polyfit_n + i] = 1;
    }
    for (auto j = 0, d = 0; j < polyfit_n; j++) {
        for (auto i = j, b = d; i < polyfit_n; i++) {
            auto temp = 0.0;
            auto a = 0, c = j;
            while (a < j) {
                temp += L[b + a] * U[c];
                c += polyfit_n;
                a++;
            }
            L[b + j] = Kmul[b + j] - temp;
            b += polyfit_n;
        }
        auto i = j + 1;
        while (i < polyfit_n) {
            auto temp = 0.0;
            auto a = 0, c = i;
            while (a < j) {
                temp += L[d + a] * U[c];
                a++;
                c += polyfit_n;
            }
            if (L[d + j] == 0) {
                return false;
            }
            U[d + i] = (Kmul[d + i] - temp) / L[d + j];
            i++;
        }
        d += polyfit_n;
    }
    return true;
}

bool PolyFit::matrix_inv(void)
{
    if (!matrix_lu()) {
        return false;
    }

    auto a = polyfit_n * polyfit_n;
    for (auto i = 0; i < polyfit_n; i++) {
        ix[i] = id[i] = 0;
    }
    for (auto i = 0; i < polyfit_n; i++) {
        auto j = 0;
        for (j = 0; j < polyfit_n; j++) {
            ie[j] = 0;
        }
        ie[i] = 1;
        j = 0;
        auto b = 0;
        while (j < polyfit_n) {
            auto temp = 0.0;
            a = 0;
            while (a < j) {
                temp += id[a] * L[b + a];
                a++;
            }
            id[j] = ie[j] - temp;
            id[j] /= L[b + j];
            j++;
            b += polyfit_n;
        }
        j = polyfit_n - 1;
        b -= polyfit_n;
        while (j > -1) {
            auto temp = 0.0;
            a = j + 1;
            while (a < polyfit_n) {
                temp += U[b + a] * ix[a];
                a++;
            }
            ix[j] = id[j] - temp;
            ix[j] /= U[b + j];
            j--;
            b -= polyfit_n;
        }
        for (j = 0, b = i; j < polyfit_n; j++) {
            Kinv[b] = ix[j];
            b += polyfit_n;
        }
    }
    return true;
}

void PolyFit::matrix_mul_a(void)
{
    for (auto i = 0; i < polyfit_n; i++) {
        for (auto j = 0; j < polyfit_n; j++) {
            const auto b = i * polyfit_n + j;
            Kmul[b] = 0;
            for (auto k = 0; k < polyfit_n; k++) {
                Kmul[b] += KT[i * polyfit_n + k] * x2[k * polyfit_n + j];
            }
        }
    }
}

void PolyFit::matrix_mul_b(void)
{
    for (auto i = 0; i < polyfit_n; i++) {
        Kb[i] = 0;
        for (auto k = 0; k < polyfit_n; k++) {
            Kb[i] += KT[i * polyfit_n + k] * y2[k];
        }
    }
}

void PolyFit::matrix_mul_c(void)
{
    for (auto i = 0; i < polyfit_n; i++) {
        ks[i] = 0;
        for (auto k = 0; k < polyfit_n; k++) {
            ks[i] += Kinv[i * polyfit_n + k] * Kb[k];
        }
    }
}

void PolyFit::matrix_t(void)
{
    for (auto i = 0, a = 0; i < polyfit_n; i++) {
        for (auto j = 0, b = 0; j < polyfit_n; j++) {
            KT[b + i] = x2[a + j];
            b += polyfit_n;
        }
        a += polyfit_n;
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

std::tuple<double, double> PolyFit::calculate_direction(std::array<double, max_length> price_steps, int degree, int length)
{
    this->degree = degree;
    this->length = length;
    polyfit_n = degree + 1;

    for (auto i = 0, idx = 0; i < polyfit_n; i++) {
        y2[i] = 0;
        for (auto j = 0; j < polyfit_n; j++) {
            x2[idx + j] = 0;
        }
        idx += polyfit_n;
    }

    x2[0] = length;
    for (auto i = 0; i < polyfit_n; i++) {
        for (auto j = i + 1; j < polyfit_n; j++) {
            auto temp = 0.0;
            auto idx = 0;
            for (idx = 0; idx < length; idx++) {
                temp += std::pow(x[idx], i + j);
            }
            idx = j;
            for (auto n = i; n < polyfit_n; n++) {
                if (idx >= 0) {
                    x2[n * polyfit_n + idx] = temp;
                }
                idx--;
            }
        }
    }

    auto temp = 0.0;
    for (auto i = 0; i < length; i++) {
        temp += std::pow(x[i], polyfit_n * 2 - 2);
    }
    x2[polyfit_n * polyfit_n - 1] = temp;
    for (auto i = 0; i < polyfit_n; i++) {
        temp = 0;
        for (auto j = 0; j < length; j++) {
            temp += price_steps[max_length - length + j] * std::pow(x[j], i);
        }
        y2[i] = temp;
    }

    if (matrix_solve()) {
        // Todo: calculate direction from ks

        auto p0 = ks.at(0);
        auto p1 = ks.at(0);
        for (auto i = 1; i < polyfit_n; i++) {
            p0 += ks[i] * std::pow(x[length - 2], i);
            p1 += ks[i] * std::pow(x[length - 1], i);
        }

        return { p0, p1 };
    }

    return { 0.0, 0.0 };
}
