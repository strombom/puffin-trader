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
    std::string_view symbol,
    std::string_view path,
    const sptrBinanceKlines klines,
    const sptrIndicators indicators,
    time_point_ms timestamp_start,
    time_point_ms timestamp_end,
    std::shared_ptr < std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>> ground_truth,
    std::shared_ptr < std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>> ground_truth_timestamps
) {
    
    std::filesystem::create_directories(path);
    auto file_path = std::string{ path } + "/" + date::format("%F", timestamp_start) + "_" + std::string{ symbol } + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"ind_idx\"";
    csv_file << ",\"timestamp\"";
    //csv_file << ",\"delta\"";

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

    for (auto gt_idx = 0; gt_idx < ground_truth->size(); gt_idx++) {
        if (indicators->timestamps[gt_idx] < timestamp_start) {
            continue;
        }
        else if (indicators->timestamps[gt_idx] > timestamp_end) {
            break;
        }

        for (auto profit_idx = 0; profit_idx < BitBot::TrainingData::take_profit.size(); profit_idx++) {
            if (ground_truth->at(gt_idx).at(profit_idx) == 0) {
                goto end_of_loop;
            }

            if (ground_truth_timestamps->at(gt_idx).at(profit_idx) < timestamp_start) {
                goto end_of_loop;
            }

            //if (suffix == "train" && ground_truth_timestamps->at(gt_idx).at(profit_idx) > timestamp_end) {
            //    goto end_of_loop;
            //}
        }

        const int indicator_idx = gt_idx; // gt_idx - (BitBot::Indicators::max_length - 1);

        csv_file << indicator_idx;
        csv_file << ",\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(indicator_idx)) << "\"";
        //csv_file << "," << intrinsic_events->events[gt_idx].delta;

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

void save_indices(std::string_view path, std::string_view filename, const std::vector<int> &indices) {
    std::filesystem::create_directories(path);
    auto file_path = std::string{ path } + "/" + std::string{ filename };
    auto csv_file = std::ofstream{ file_path, std::ios::binary };
    csv_file << "\"ind_idx\"\n";
    for (const auto& idx : indices) {
        csv_file << idx << "\n";
    }
    csv_file.close();
}

void TrainingData::make_sections(std::string_view path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start)
{
    std::filesystem::create_directories(path);

    if (true) {
        if (ground_truth_symbol != symbol) {
            for (auto& thread : threads) {
                thread.join();
            }
            threads.clear();
            make_ground_truth(symbol, klines, indicators);
        }
        //save_ground_truth(symbol);
    }
    else {
        load_ground_truth(symbol);
    }

    const auto profit_idx = BitBot::TrainingData::take_profit.size() - 1;
    const auto data_length = indicators->indicators.size();
    auto timestamp_train_start = timestamp_start + date::days{ 2 };
    auto train_start_idx = 0;
     
    while (true) {
        const auto timestamp_train_end = timestamp_train_start + BitBot::history_length;
        const auto timestamp_val_start = timestamp_train_end;
        const auto timestamp_val_end = timestamp_val_start + date::days{ 1 };

        // Make training data
        while (indicators->timestamps[train_start_idx] < timestamp_train_start) {
            train_start_idx++;
            if (train_start_idx == data_length) {
                goto end_of_indicators;
            }
        }
        if (indicators->timestamps[train_start_idx].time_since_epoch().count() == 0) {
            goto end_of_indicators;
        }

        auto training_indices = std::vector<int>{};
        auto indicator_idx = train_start_idx;
        while (indicators->timestamps[indicator_idx] < timestamp_train_end) {
            const auto gt_max_timestamp = (*ground_truth_timestamps)[indicator_idx][profit_idx];
            if (gt_max_timestamp < timestamp_train_end && gt_max_timestamp.time_since_epoch().count() > 0) {
                training_indices.push_back(indicator_idx);
            }

            indicator_idx++;
            if (indicator_idx == data_length) {
                goto end_of_indicators;
            }
        }

        // Make validation data
        while (indicators->timestamps[indicator_idx] < timestamp_val_start) {
            indicator_idx++;
            if (indicator_idx == data_length) {
                goto end_of_indicators;
            }
        }

        auto validation_indices = std::vector<int>{};
        while (indicators->timestamps[indicator_idx] < timestamp_val_end) {
            const auto gt_max_timestamp = (*ground_truth_timestamps)[indicator_idx][profit_idx];
            if (gt_max_timestamp < timestamp_val_end && gt_max_timestamp.time_since_epoch().count() > 0) {

                validation_indices.push_back(indicator_idx);
            }

            indicator_idx++;
            if (indicator_idx == data_length) {
                goto end_of_indicators;
            }
        }

        save_indices(path, "/" + date::format("%F", timestamp_train_start) + "_train_" + std::string{ symbol } + ".csv", training_indices);
        save_indices(path, "/" + date::format("%F", timestamp_val_start) + "_val_" + std::string{ symbol } + ".csv", validation_indices);

        timestamp_train_start += date::days{ 1 };
    }

end_of_indicators:
    ;
}

void TrainingData::make(std::string_view path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, klines, indicators);
    make_section_thread(symbol, path, klines, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps);
}
 
void TrainingData::make_ground_truth(std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators)
{
    if (ground_truth_symbol != symbol) {
        ground_truth->clear();
        ground_truth_timestamps->clear();
        ground_truth_symbol = symbol;
    }

    ground_truth->resize(indicators->timestamps.size());
    ground_truth_timestamps->resize(indicators->timestamps.size());

    auto positions = std::array<std::list<Position>, BitBot::TrainingData::take_profit.size()>{};
    auto ind_idx = 0;

    std::array<int, BitBot::TrainingData::take_profit.size()> maxcount = { 0 };

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
                (*ground_truth)[position->ind_idx][profit_idx] = 1;
                (*ground_truth_timestamps)[position->ind_idx][profit_idx] = indicators->timestamps[ind_idx];
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
                    (*ground_truth)[position->ind_idx][profit_idx] = -1;
                    (*ground_truth_timestamps)[position->ind_idx][profit_idx] = indicators->timestamps[ind_idx];
                    position->remove = true;
                    remove = true;
                }
            }

            if (remove) {
                positions[profit_idx].remove_if([](const Position& position) { return position.remove; });
            }
        }

        while (klines->rows[kline_idx].timestamp >= indicators->timestamps[ind_idx] && indicators->timestamps.size() > ind_idx + 1) {
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
                    ind_idx,
                    take_profit,
                    stop_loss
                });
            }

            ind_idx++;
        }
    }

    printf("Maxcount %s:", std::string{ symbol }.c_str());
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        printf("   (%d) %d", idx, maxcount[idx]);
    }
    printf("\n");
}

void TrainingData::save_ground_truth(std::string_view symbol)
{
    auto file_path = std::string{ BitBot::path } + "\\ground_truth";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + std::string{ symbol } + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };

    for (auto row_idx = 0; row_idx < ground_truth->size(); row_idx++) {
        for (auto col_idx = 0; col_idx < BitBot::TrainingData::take_profit.size(); col_idx++) {
            const auto gt_value = ground_truth->at(row_idx)[col_idx];
            const auto gt_timestamp = ground_truth_timestamps->at(row_idx)[col_idx];
            data_file.write(reinterpret_cast<const char*>(&gt_value), sizeof(gt_value));
            data_file.write(reinterpret_cast<const char*>(&gt_timestamp), sizeof(gt_timestamp));
        }
    }

    data_file.close();
}

void TrainingData::load_ground_truth(std::string_view symbol)
{
    const auto file_path = std::string{ BitBot::path } + "\\ground_truth\\" + std::string{ symbol } + ".dat";

    const auto row_size = (sizeof(int) + sizeof(time_point_ms)) * BitBot::TrainingData::take_profit.size();
    const auto gt_length = std::filesystem::file_size(file_path) / row_size;

    auto data_file = std::ifstream{ file_path, std::ios::binary };
    auto intrinsic_event = IntrinsicEvent{};

    ground_truth->resize(gt_length);
    ground_truth_timestamps->resize(gt_length);

    for (auto row_idx = 0; row_idx < gt_length; row_idx++) {
        for (auto col_idx = 0; col_idx < BitBot::TrainingData::take_profit.size(); col_idx++) {
            int gt_value;
            time_point_ms timestamp;
            data_file.read(reinterpret_cast <char*> (&gt_value), sizeof(gt_value));
            data_file.read(reinterpret_cast <char*> (&timestamp), sizeof(timestamp));

            (*ground_truth)[row_idx][col_idx] = gt_value;
            (*ground_truth_timestamps)[row_idx][col_idx] = timestamp;
        }
    }
    data_file.close();
}
