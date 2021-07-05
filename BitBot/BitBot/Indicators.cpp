#include "pch.h"

#include "Indicators.h"
#include "PolyFit.h"
#include "BitLib/Logger.h"

#include <filesystem>


Indicators::Indicators(std::string symbol) : symbol(symbol) 
{
    load();
}

void Indicators::calculate(const sptrIntrinsicEvents intrinsic_events)
{
    //n_steps = (int)intrinsic_events->events.size();

    auto price_steps = std::array<double, max_length>{};
    for (auto i = 0; i < max_length; i++) {
        price_steps[i] = intrinsic_events->events[i].price;
    }

    for (auto idx = max_length - 1; idx < intrinsic_events->events.size(); idx++) {
        price_steps[max_length - 1] = intrinsic_events->events[idx].price;



        std::rotate(price_steps.begin(), price_steps.begin() + 1, price_steps.end());
    }







    /*
    auto price_steps = std::array<double, BitBot::Indicators::max_length>{};

    for (auto degree_idx = 0; degree_idx < BitBot::Indicators::degrees.size(); degree_idx++) {
        const auto degree = BitBot::Indicators::degrees[degree_idx];

        for (auto length_idx = 0; length_idx < BitBot::Indicators::lengths.size(); length_idx++) {
            const auto length = BitBot::Indicators::lengths[length_idx];

            const auto indicator_idx = degree_idx * BitBot::Indicators::lengths.size() * 2 + length_idx * 2;
            logger.info("Indicators::calculate (%s) degree_idx(%d) length_idx(%d) indicator_idx(%d)", symbol.c_str(), degree_idx, length_idx, indicator_idx);

            auto poly_fit = PolyFit{degree, length};

            for (auto idx = max_length - 1; idx < intrinsic_events->events.size(); idx++) {
                for (auto i = 0; i < length; i++) {
                    price_steps[i] = intrinsic_events->events[idx - length + i + 1].price;
                }

                const auto [p0, p1] = poly_fit.calculate_direction(price_steps);
                const auto price = intrinsic_events->events[idx].price;
                const auto price_diff = p1 / price - 1.0;
                const auto direction = p1 / p0 - 1.0;



                indicators->at(idx).at(indicator_idx + 0) = (float)price_diff;
                indicators->at(idx).at(indicator_idx + 1) = (float)direction;
            }
        }
    }
    */
}

std::ostream& operator<<(std::ostream& stream, const Indicators& indicators)
{
    for (auto ts = 0; ts < indicators.indicators.size(); ts++) {
        for (auto indx = 0; indx < BitBot::Indicators::indicator_width; indx++) {
            stream.write(reinterpret_cast<const char*>(&indicators.indicators.at(ts).at(indx)), sizeof(float));
        }
    }

    return stream;
}

std::istream& operator>>(std::istream& stream, Indicators& indicators)
{
    auto ts = 0;
    auto indx = 0;

    auto tmp_indicator = std::array<float, BitBot::Indicators::indicator_width>{};

    while (true) {
        float value;
        stream.read(reinterpret_cast <char*> (&value), sizeof(value));
        if (!stream) {
            break;
        }
        tmp_indicator.at(indx) = value;
        
        indx = (indx + 1) % BitBot::Indicators::indicator_width;
        if (indx == 0) {
            indicators.indicators.push_back(tmp_indicator);
            ts++;
        }
    }

    return stream;
}

void Indicators::load(void)
{
    const auto file_path = std::string{ BitBot::path } + "\\indicators\\" + symbol + ".dat";
    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        data_file >> *this;
        data_file.close();
    }
}

void Indicators::save(void) const
{
    auto file_path = std::string{ BitBot::path } + "\\indicators";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };
    data_file << *this;
    data_file.close();
}
