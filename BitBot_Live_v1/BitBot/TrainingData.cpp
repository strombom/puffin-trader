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

void make_section_thread(
    const std::string& symbol,
    const std::string& suffix,
    const std::string& path,
    const sptrBinanceKlines klines,
    const sptrIntrinsicEvents intrinsic_events,
    const sptrIndicators indicators,
    time_point_ms timestamp_start,
    time_point_ms timestamp_end,
    std::shared_ptr < std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>> ground_truth,
    std::shared_ptr < std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>> ground_truth_timestamps
) {

    std::filesystem::create_directories(path);
    auto file_path = path + "\\" + date::format("%F", timestamp_start) + "_" + symbol + "_" + suffix + ".csv";

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

void TrainingData::make_section(const std::string& path, const std::string& symbol, const std::string& suffix, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    if (ground_truth_symbol != symbol) {
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        make_ground_truth(symbol, klines, intrinsic_events);
    }

    auto thread = std::thread{ make_section_thread, symbol, suffix, path, klines, intrinsic_events, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps };
    threads.push_back(std::move(thread));

    if (threads.size() == (int) (std::thread::hardware_concurrency() * 1.3)) {
        threads.begin()->join();
        threads.erase(threads.begin());
    }
}

void TrainingData::make(const std::string& path, const std::string& symbol, const sptrBinanceKlines klines, const sptrIntrinsicEvents intrinsic_events, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, klines, intrinsic_events);
    make_section_thread(symbol, "sim", path, klines, intrinsic_events, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps);
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

    auto positions = std::array<std::list<Position>, BitBot::TrainingData::take_profit.size()>{};
    auto ie_idx = 0;

    std::array<int, 9> maxcount = { 0 };

    for (auto kline_idx = 0; kline_idx < klines->rows.size(); kline_idx++) {
        const auto mark_price = klines->rows[kline_idx].open;

        auto remove = false;

        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            maxcount[profit_idx] = std::max(maxcount[profit_idx], (int)positions[profit_idx].size());

            auto position = positions[profit_idx].begin();
            while (position != positions[profit_idx].end()) {
                if (mark_price < position->take_profit) {
                    break;
                }
                (*ground_truth)[position->ie_idx][profit_idx] = 1;
                (*ground_truth_timestamps)[position->ie_idx][profit_idx] = intrinsic_events->events[ie_idx].timestamp;
                position->remove = true;
                remove = true;

                position = std::next(position);
            }

            if (positions[profit_idx].size() > 0) {
                position = positions[profit_idx].end();
                while (position != positions[profit_idx].begin()) {
                    position = std::prev(position);
                    if (mark_price > position->stop_loss) {
                        break;
                    }
                    (*ground_truth)[position->ie_idx][profit_idx] = -1;
                    (*ground_truth_timestamps)[position->ie_idx][profit_idx] = intrinsic_events->events[ie_idx].timestamp;
                    position->remove = true;
                    remove = true;
                }
            }

            if (remove) {
                positions[profit_idx].remove_if([](const Position& position) { return position.remove; });
            }
        }

        while (klines->rows[kline_idx].timestamp >= intrinsic_events->events[ie_idx].timestamp && intrinsic_events->events.size() > ie_idx + 1) {
            /*
            printf(
                "while kline (%d) %s   event (%d) %s\n", 
                kline_idx,
                DateTime::to_string_iso_8601(klines->rows[kline_idx].timestamp).c_str(), 
                ie_idx,
                DateTime::to_string_iso_8601(intrinsic_events->events[ie_idx].timestamp).c_str()
            );
            */

            //const auto mark_price = intrinsic_events->events.at(ie_idx).price;

            for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
                const auto take_profit = (float)(mark_price * BitBot::TrainingData::take_profit[profit_idx]);
                const auto stop_loss = (float)(mark_price * BitBot::TrainingData::stop_loss[profit_idx]);

                //auto pos_idx = 0;
                auto position = positions[profit_idx].begin();
                for (; position != positions[profit_idx].end(); ++position) {
                    if (position->take_profit > take_profit) {
                        break;
                    }
                }
                positions[profit_idx].insert(position, {
                    ie_idx,
                    take_profit,
                    stop_loss
                });
            }

            ie_idx++;
        }
    }

    printf("Maxcount %s:", symbol.c_str());
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        printf("   (%d) %d", idx, maxcount[idx]);
    }
    printf("\n");
}
