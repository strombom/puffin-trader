#include "pch.h"

#include "TrainingData.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


TrainingData::TrainingData()
{

}

void TrainingData::make(const std::string& symbol, const sptrBinanceKlines binance_klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators)
{
    auto ground_truth = make_ground_truth(intrinsic_events);

    auto file_path = std::string{ BitBot::path } + "\\training_data";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    for (auto&& degree : BitBot::Indicators::degrees) {
        for (auto&& length : BitBot::Indicators::lengths) {
            csv_file << "\"" << std::to_string(degree) << "-" << std::to_string(length) << "-p\",";
            csv_file << "\"" << std::to_string(degree) << "-" << std::to_string(length) << "-d\",";
        }
    }

    auto symbols_string = std::string{};
    for (auto&& a_symbol : BitBot::symbols) {
        csv_file << "\"" << std::string{a_symbol} + "\",";
        if (a_symbol == symbol) {
            symbols_string += "True,";
        }
        else {
            symbols_string += "False,";
        }
    }

    csv_file.precision(2);
    csv_file << std::fixed << "\"(" << BitBot::TrainingData::take_profit << "," << BitBot::TrainingData::stop_loss << ")\"\n";
    csv_file << std::defaultfloat;

    // indicators:    259803 = ground_truth - (max_length - 1)
    // ground_truth : 259952

    for (auto idx = BitBot::Indicators::max_length - 1; idx < ground_truth.size(); idx++) {

        if (ground_truth.at(idx) == 0) {
            break;
        }
        const auto& indicator = indicators->indicators.at(idx - (BitBot::Indicators::max_length - 1));
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << indicator.at(i) << ",";
        }
        csv_file << symbols_string;
        csv_file << ground_truth.at(idx) << "\n";
    }

    csv_file.close();
}
 
std::vector<int> TrainingData::make_ground_truth(const sptrIntrinsicEvents intrinsic_events)
{
    auto ground_truth = std::vector<int>{};
    ground_truth.resize(intrinsic_events->events.size());

    auto positions = std::list<Position>{};

    for (auto idx = 0; idx < intrinsic_events->events.size(); idx++) {
        const auto mark_price = intrinsic_events->events.at(idx).price;

        auto remove = false;
        for (auto&& position : positions) {
            if (mark_price >= position.take_profit) {
                ground_truth.at(position.idx) = 1;
                position.remove = true;
                remove = true;
            }
            else if (mark_price <= position.stop_loss) {
                ground_truth.at(position.idx) = -1;
                position.remove = true;
                remove = true;
            }
        }

        if (remove) {
            positions.remove_if([](const Position& position) { return position.remove; });
        }

        positions.emplace_back(
            idx, 
            (float)(mark_price * BitBot::TrainingData::take_profit), 
            (float)(mark_price * BitBot::TrainingData::stop_loss)
        );
    }

    return ground_truth;
}
