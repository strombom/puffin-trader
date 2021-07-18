#include "pch.h"

#include "TrainingData.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


TrainingData::TrainingData()
{

}

void TrainingData::make_section(const std::string& symbol, const std::string& suffix, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, intrinsic_events);

    auto file_path = std::string{ BitBot::path } + "\\training_data_sections";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + date::format("%F", timestamp_start) + "_" + symbol + "_" + suffix + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"timestamp\"";

    for (auto&& degree : BitBot::Indicators::degrees) {
        for (auto&& length : BitBot::Indicators::lengths) {
            csv_file << "\"" << std::to_string(degree) << "-" << std::to_string(length) << "-p\",";
            csv_file << "\"" << std::to_string(degree) << "-" << std::to_string(length) << "-d\",";
        }
    }

    auto symbols_string = std::string{};
    for (auto&& a_symbol : BitBot::symbols) {
        csv_file << "\"" << std::string{ a_symbol } + "\",";
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

    for (auto gt_idx = BitBot::Indicators::max_length - 1; gt_idx < ground_truth.size(); gt_idx++) {
        if (ground_truth.at(gt_idx) == 0) {
            break;
        }

        if (intrinsic_events->events[gt_idx].timestamp < timestamp_start) {
            continue;
        }

        if (intrinsic_events->events[gt_idx].timestamp > timestamp_end) {
            break;
        }

        if (ground_truth_timestamps.at(gt_idx) < timestamp_start) {
            break;
        }

        if (suffix != "valid" && ground_truth_timestamps.at(gt_idx) > timestamp_end) {
            break;
        }

        const int indicator_idx = gt_idx - (BitBot::Indicators::max_length - 1);

        csv_file << "\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(indicator_idx)) << "\",";

        const auto& indicator = indicators->indicators.at(indicator_idx);
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << indicator.at(i) << ",";
        }
        csv_file << symbols_string;
        csv_file << ground_truth.at(gt_idx) << "\n";
    }

    csv_file.close();
}

void TrainingData::make(const std::string& symbol, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, intrinsic_events);

    auto file_path = std::string{ BitBot::path } + "\\training_data";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"date\",";

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

    for (auto gt_idx = BitBot::Indicators::max_length - 1; gt_idx < ground_truth.size(); gt_idx++) {

        if (ground_truth.at(gt_idx) == 0) {
            break;
        }

        if (intrinsic_events->events[gt_idx].timestamp < timestamp_start) {
            continue;
        }

        if (intrinsic_events->events[gt_idx].timestamp > timestamp_end) {
            break;
        }

        csv_file << DateTime::to_string_iso_8601(indicators->timestamps.at(gt_idx - (BitBot::Indicators::max_length - 1))) << ",";

        const auto& indicator = indicators->indicators.at(gt_idx - (BitBot::Indicators::max_length - 1));
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << indicator.at(i) << ",";
        }
        csv_file << symbols_string;
        csv_file << ground_truth.at(gt_idx) << "\n";
    }

    csv_file.close();
}
 
void TrainingData::make_ground_truth(const std::string symbol, const sptrIntrinsicEvents intrinsic_events)
{
    if (ground_truth_symbol != symbol) {
        ground_truth.clear();
        ground_truth_timestamps.clear();
        ground_truth_symbol = symbol;
    }

    ground_truth.resize(intrinsic_events->events.size());
    ground_truth_timestamps.resize(intrinsic_events->events.size());

    auto positions = std::list<Position>{};

    for (auto idx = 0; idx < intrinsic_events->events.size(); idx++) {
        const auto mark_price = intrinsic_events->events.at(idx).price;

        auto remove = false;
        for (auto&& position : positions) {
            if (mark_price >= position.take_profit) {
                ground_truth.at(position.idx) = 1;
                ground_truth_timestamps.at(position.idx) = intrinsic_events->events.at(idx).timestamp;
                position.remove = true;
                remove = true;
            }
            else if (mark_price <= position.stop_loss) {
                ground_truth.at(position.idx) = -1;
                ground_truth_timestamps.at(position.idx) = intrinsic_events->events.at(idx).timestamp;
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
}
