#include "pch.h"

#include "PolyFit.h"

#include <filesystem>
#include <fstream>



PolyFit::PolyFit(void)
{

}

std::tuple<double, double> PolyFit::calculate_direction(std::array<double, max_length> price_steps, int degree, int length)
{
    // https://gist.github.com/chrisengelsma/108f7ab0a746323beaaf7d6634cf4add

    const auto degree_p1 = degree + 1;
    const auto degree_t2p1 = degree * 2 + 1;

    // X = sigma(xi^2n)
    for (int i = 0; i < degree_t2p1; ++i) {
        X[i] = 0;
        for (int x = 0; x < length; ++x) {
            X[i] += std::pow(x, i);
        }
    }

    // B = normal augmented matrix that stores the equations
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            B[i][j] = X[i + j];
        }
    }

    // Y = sigma(xi^n * yi)
    for (int i = 0; i < degree_p1; ++i) {
        Y[i] = 0.0;
        for (int x = 0; x < length; ++x) {
            Y[i] += std::pow(x, i) * price_steps[max_length - length + x];
        }
    }

    // Load values of Y as last column of B
    for (int i = 0; i <= degree; ++i) {
        B[i][degree_p1] = Y[i];
    }

    // Pivotisation of the B matrix.
    for (int i = 0; i < degree_p1; ++i) {
        for (int k = i + 1; k < degree_p1; ++k) {
            if (B[i][i] < B[k][i]) {
                for (int j = 0; j <= degree_p1; ++j) {
                    const auto tmp = B[i][j];
                    B[i][j] = B[k][j];
                    B[k][j] = tmp;
                }
            }
        }
    }

    // Performs the Gaussian elimination.
    for (int i = 0; i < degree; ++i) {
        for (int k = i + 1; k < degree_p1; ++k) {
            const auto t = B[k][i] / B[i][i];
            for (int j = 0; j <= degree_p1; ++j) {
                // Make all elements below the pivot equals to zero
                //  or eliminate the variable.
                B[k][j] -= t * B[i][j];
            }
        }
    }

    // Back substitution.
    for (int i = degree; i >= 0; --i) {
        // Set the variable as the rhs of last equation
        coeffs[i] = B[i][degree_p1];
        for (int j = 0; j < degree_p1; ++j) {
            if (j != i) {
                // Subtract all lhs values except the target coefficient.
                coeffs[i] -= B[i][j] * coeffs[j];
            }
        }
        // Divide rhs by coefficient of variable being calculated.
        coeffs[i] /= B[i][i];
    }

    /*
    auto file_path = std::string{ BitBot::path } + "\\polyfit";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + std::to_string(degree) + "-" + std::to_string(length) + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"x\";\"p\";\"y\"\n";

    for (auto x = 0; x < length; x++) {
        csv_file << x << ";";
        csv_file << price_steps[max_length - length + x] << ";";

        auto y = 0.0;
        for (auto i = 0; i < degree_p1; i++) {
            y += coeffs[i] * std::pow(x, i);
        }
        csv_file << y << "\n";
    }

    csv_file.close();
    */

    // Result
    auto p0 = 0.0;
    auto p1 = 0.0;
    for (auto i = 0; i < degree_p1; i++) {
        p0 += coeffs[i] * std::pow(length - 2, i);
        p1 += coeffs[i] * std::pow(length - 1, i);
    }

    return { p0, p1 };
}
