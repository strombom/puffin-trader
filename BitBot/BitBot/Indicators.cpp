#include "pch.h"

#include "Indicators.h"
#include "PolyFit.h"
#include "BitLib/Logger.h"

#include <filesystem>


Indicators::Indicators(std::string symbol)
{
    load(symbol);
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
            time_point_ms timestamp;
            stream.read(reinterpret_cast <char*> (&timestamp), sizeof(timestamp));
            if (!stream) {
                printf("Error, read indicators!\n");
                break;
            }
            indicators.timestamps.push_back(timestamp);
            ts++;
        }
    }

    return stream;
}

void Indicators::load(std::string symbol)
{
    const auto file_path = std::string{ BitBot::path } + "\\indicators\\" + symbol + ".dat";
    if (std::filesystem::exists(file_path)) {
        auto data_file = std::ifstream{ file_path, std::ios::binary };
        data_file >> *this;
        data_file.close();
    }
}

/*
void Indicators::save(void) const
{
    auto file_path = std::string{ BitBot::path } + "\\indicators";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };
    data_file << *this;
    data_file.close();
}
*/

void calculate_thread(const std::string symbol, const sptrIntrinsicEvents intrinsic_events)
{
    logger.info("Indicators calculate %s", symbol.c_str());

    std::vector<std::array<float, BitBot::Indicators::indicator_width>> indicators;
    std::vector<time_point_ms> timestamps;

    //if (indicators.size() == intrinsic_events->events.size() - max_length + 1) {
    //    return;
    //}
    //else {
    //indicators.clear();
    //timestamps.clear();
    indicators.resize(intrinsic_events->events.size() - max_length + 1);
    timestamps.resize(intrinsic_events->events.size() - max_length + 1);
    //}

    auto price_steps = std::array<double, max_length>{};
    for (auto i = 0; i < max_length; i++) {
        price_steps[i] = intrinsic_events->events[i].price;
    }

    auto poly_fit = PolyFit{ };
    auto tmp_indicator = std::array<float, BitBot::Indicators::indicator_width>{};

    for (auto ie_idx = max_length - 1; ie_idx < intrinsic_events->events.size(); ie_idx++) {
        price_steps[max_length - 1] = intrinsic_events->events[ie_idx].price;

        auto indicator_idx = 0;
        for (auto degree_idx = 0; degree_idx < BitBot::Indicators::degrees.size(); degree_idx++) {
            const auto degree = BitBot::Indicators::degrees[degree_idx];

            for (auto length_idx = 0; length_idx < BitBot::Indicators::lengths.size(); length_idx++) {
                const auto length = BitBot::Indicators::lengths[length_idx];

                const auto [p0, p1] = poly_fit.calculate_direction(price_steps, degree, length);
                const auto price_diff = p1 / intrinsic_events->events[ie_idx].price - 1.0;
                const auto direction = p1 / p0 - 1.0;

                tmp_indicator[indicator_idx + 0] = (float)price_diff;
                tmp_indicator[indicator_idx + 1] = (float)direction;
                indicator_idx += 2;
            }
        }

        indicators.at(ie_idx - max_length + 1) = tmp_indicator;
        timestamps.at(ie_idx - max_length + 1) = intrinsic_events->events[ie_idx].timestamp;;
        std::rotate(price_steps.begin(), price_steps.begin() + 1, price_steps.end());
    }

    auto file_path = std::string{ BitBot::path } + "\\indicators";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };

    for (auto ts = 0; ts < indicators.size(); ts++) {
        for (auto indx = 0; indx < BitBot::Indicators::indicator_width; indx++) {
            data_file.write(reinterpret_cast<const char*>(&indicators[ts][indx]), sizeof(float));
        }
        data_file.write(reinterpret_cast<const char*>(&timestamps[ts]), sizeof(time_point_ms));
    }

    data_file.close();
}

void Indicators::calculate_and_save(std::string symbol, sptrIntrinsicEvents intrinsic_events)
{
    auto thread = std::thread{ calculate_thread, symbol, intrinsic_events };
    threads.push_back(std::move(thread));

    if (threads.size() == (int)(std::thread::hardware_concurrency() * 0.8)) {
        threads.begin()->join();
        threads.erase(threads.begin());
    }
}

void Indicators::join(void)
{
    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();
}
