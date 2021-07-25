#include "pch.h"

#include "TrainingData.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


TrainingData::TrainingData()
{
    ground_truth = std::make_shared<std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>>();
    ground_truth_timestamps = std::make_shared<std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>>();
}

void make_section_Thread(
    const std::string& symbol, 
    const std::string& suffix, 
    const sptrBinanceKlines klines, 
    const sptrIntrinsicEvents intrinsic_events, 
    const sptrIndicators indicators, 
    time_point_ms timestamp_start, 
    time_point_ms timestamp_end,
    std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>> ground_truth,
    std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>> ground_truth_timestamps
)
{

    /*
    struct Position;
    ;
    ;
    std::string ground_truth_symbol;
    */

    auto file_path = std::string{ BitBot::path } + "\\training_data_sections";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + date::format("%F", timestamp_start) + "_" + symbol + "_" + suffix + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"timestamp\"";
    csv_file << ",\"delta\"";

    for (auto&& degree : BitBot::Indicators::degrees) {
        for (auto&& length : BitBot::Indicators::lengths) {
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-p\"";
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-d\"";
        }
    }

    auto symbols_string = std::string{};
    for (auto&& a_symbol : BitBot::symbols) {
        csv_file << ",\"" << std::string{ a_symbol } + "\"";
        if (a_symbol == symbol) {
            symbols_string += ",True";
        }
        else {
            symbols_string += ",False";
        }
    }

    csv_file.precision(3);
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        csv_file << std::fixed << ",\"(" << BitBot::TrainingData::take_profit.at(idx) << "," << BitBot::TrainingData::stop_loss.at(idx) << ")\"";
    }
    csv_file << "\n";
    csv_file << std::defaultfloat;

    // indicators:    259803 = ground_truth - (max_length - 1)
    // ground_truth : 259952

    for (auto gt_idx = BitBot::Indicators::max_length - 1; gt_idx < ground_truth.size(); gt_idx++) {
        if (intrinsic_events->events[gt_idx].timestamp < timestamp_start) {
            continue;
        }

        if (intrinsic_events->events[gt_idx].timestamp > timestamp_end) {
            break;
        }

        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            if (ground_truth.at(gt_idx).at(profit_idx) == 0) {
                goto end_of_loop;
            }

            if (ground_truth_timestamps.at(gt_idx).at(profit_idx) < timestamp_start) {
                goto end_of_loop;
            }

            if (suffix == "train" && ground_truth_timestamps.at(gt_idx).at(profit_idx) > timestamp_end) {
                goto end_of_loop;
            }
        }

        const int indicator_idx = gt_idx - (BitBot::Indicators::max_length - 1);

        csv_file << "\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(indicator_idx)) << "\"";
        csv_file << "," << intrinsic_events->events[gt_idx].delta;

        const auto& indicator = indicators->indicators.at(indicator_idx);
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << "," << indicator.at(i);
        }
        csv_file << symbols_string;
        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            csv_file << "," << ground_truth.at(gt_idx).at(profit_idx);
        }
        csv_file << "\n";
    }

end_of_loop:
    csv_file.close();
}

void little_thread(
    const std::string& symbol,
    const std::string& suffix,
    const sptrBinanceKlines klines,
    const sptrIntrinsicEvents intrinsic_events,
    const sptrIndicators indicators,
    time_point_ms timestamp_start,
    time_point_ms timestamp_end,
    std::shared_ptr < std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>> ground_truth,
    std::shared_ptr < std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>> ground_truth_timestamps
) {

    auto file_path = std::string{ BitBot::path } + "\\training_data_sections";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + date::format("%F", timestamp_start) + "_" + symbol + "_" + suffix + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"timestamp\"";
    csv_file << ",\"delta\"";

    for (auto&& degree : BitBot::Indicators::degrees) {
        for (auto&& length : BitBot::Indicators::lengths) {
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-p\"";
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-d\"";
        }
    }

    auto symbols_string = std::string{};
    for (auto&& a_symbol : BitBot::symbols) {
        csv_file << ",\"" << std::string{ a_symbol } + "\"";
        if (a_symbol == symbol) {
            symbols_string += ",True";
        }
        else {
            symbols_string += ",False";
        }
    }

    csv_file.precision(3);
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        csv_file << std::fixed << ",\"(" << BitBot::TrainingData::take_profit.at(idx) << "," << BitBot::TrainingData::stop_loss.at(idx) << ")\"";
    }
    csv_file << "\n";
    csv_file << std::defaultfloat;

    // indicators:    259803 = ground_truth - (max_length - 1)
    // ground_truth : 259952

    for (auto gt_idx = BitBot::Indicators::max_length - 1; gt_idx < ground_truth->size(); gt_idx++) {
        if (intrinsic_events->events[gt_idx].timestamp < timestamp_start) {
            continue;
        }

        if (intrinsic_events->events[gt_idx].timestamp > timestamp_end) {
            break;
        }

        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            if (ground_truth->at(gt_idx).at(profit_idx) == 0) {
                goto end_of_loop;
            }

            if (ground_truth_timestamps->at(gt_idx).at(profit_idx) < timestamp_start) {
                goto end_of_loop;
            }

            if (suffix == "train" && ground_truth_timestamps->at(gt_idx).at(profit_idx) > timestamp_end) {
                goto end_of_loop;
            }
        }

        const int indicator_idx = gt_idx - (BitBot::Indicators::max_length - 1);

        csv_file << "\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(indicator_idx)) << "\"";
        csv_file << "," << intrinsic_events->events[gt_idx].delta;

        const auto& indicator = indicators->indicators.at(indicator_idx);
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << "," << indicator.at(i);
        }
        csv_file << symbols_string;
        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            csv_file << "," << ground_truth->at(gt_idx).at(profit_idx);
        }
        csv_file << "\n";
    }

end_of_loop:
    csv_file.close();
}

void TrainingData::join(void)
{
    for (auto& thread : threads) {
        thread.join();
    }
    threads.clear();
}

void TrainingData::make_section(const std::string& symbol, const std::string& suffix, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    if (ground_truth_symbol != symbol) {
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        make_ground_truth(symbol, klines, intrinsic_events);
    }

    auto thread = std::thread{ little_thread, symbol, suffix, klines, intrinsic_events, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps };
    threads.push_back(std::move(thread));

    if (threads.size() == 8) {
        threads.begin()->join();
        threads.erase(threads.begin());
    }
}

void TrainingData::make(const std::string& symbol, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, klines, intrinsic_events);

    auto file_path = std::string{ BitBot::path } + "\\training_data";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"timestamp\"";
    csv_file << ",\"delta\"";

    for (auto&& degree : BitBot::Indicators::degrees) {
        for (auto&& length : BitBot::Indicators::lengths) {
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-p\"";
            csv_file << ",\"" << std::to_string(degree) << "-" << std::to_string(length) << "-d\"";
        }
    }

    auto symbols_string = std::string{};
    for (auto&& a_symbol : BitBot::symbols) {
        csv_file << ",\"" << std::string{a_symbol} + "\"";
        if (a_symbol == symbol) {
            symbols_string += ",True";
        }
        else {
            symbols_string += ",False";
        }
    }

    csv_file.precision(3);
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        csv_file << std::fixed << ",\"(" << BitBot::TrainingData::take_profit.at(idx) << "," << BitBot::TrainingData::stop_loss.at(idx) << ")\"";
    }
    csv_file << "\n";
    csv_file << std::defaultfloat;

    // indicators:    259803 = ground_truth - (max_length - 1)
    // ground_truth : 259952

    for (auto gt_idx = BitBot::Indicators::max_length - 1; gt_idx < ground_truth->size(); gt_idx++) {
        if (intrinsic_events->events[gt_idx].timestamp < timestamp_start) {
            continue;
        }

        if (intrinsic_events->events[gt_idx].timestamp > timestamp_end) {
            break;
        }

        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            if (ground_truth->at(gt_idx).at(profit_idx) == 0) {
                goto end_of_loop;
            }
        }

        csv_file << "\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(gt_idx - (BitBot::Indicators::max_length - 1))) << "\"";
        csv_file << "," << intrinsic_events->events[gt_idx].delta;

        const auto& indicator = indicators->indicators.at(gt_idx - (BitBot::Indicators::max_length - 1));
        for (auto i = 0; i < indicator.size(); i++) {
            csv_file << "," << indicator.at(i);
        }
        csv_file << symbols_string;
        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            csv_file << "," << ground_truth->at(gt_idx).at(profit_idx);
        }
        csv_file << "\n";
    }

end_of_loop:
    csv_file.close();
}
 
void TrainingData::make_ground_truth(const std::string symbol, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events)
{
    if (ground_truth_symbol != symbol) {
        ground_truth->clear();
        ground_truth_timestamps->clear();
        ground_truth_symbol = symbol;
    }

    ground_truth->resize(intrinsic_events->events.size());
    ground_truth_timestamps->resize(intrinsic_events->events.size());

    auto positions = std::list<Position>{};

    auto ie_idx = 0;

    for (auto kline_idx = 0; kline_idx < klines->rows.size(); kline_idx++) {
        const auto mark_price = klines->rows.at(ie_idx).open;

        auto remove = false;
        for (auto&& position : positions) {
            if (mark_price >= position.take_profit) {
                ground_truth->at(position.ie_idx).at(position.profit_idx) = 1;
                ground_truth_timestamps->at(position.ie_idx).at(position.profit_idx) = intrinsic_events->events.at(ie_idx).timestamp;
                position.remove = true;
                remove = true;
            }
            else if (mark_price <= position.stop_loss) {
                ground_truth->at(position.ie_idx).at(position.profit_idx) = -1;
                ground_truth_timestamps->at(position.ie_idx).at(position.profit_idx) = intrinsic_events->events.at(ie_idx).timestamp;
                position.remove = true;
                remove = true;
            }
        }

        if (remove) {
            positions.remove_if([](const Position& position) { return position.remove; });
        }

        while (klines->rows.at(kline_idx).timestamp >= intrinsic_events->events.at(ie_idx).timestamp && intrinsic_events->events.size() > ie_idx + 1) {
            //const auto mark_price = intrinsic_events->events.at(ie_idx).price;
            
            for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
                positions.emplace_back(
                    ie_idx,
                    profit_idx,
                    (float)(mark_price * BitBot::TrainingData::take_profit.at(profit_idx)),
                    (float)(mark_price * BitBot::TrainingData::stop_loss.at(profit_idx))
                );
            }

            ie_idx++;
        }
    }
}
