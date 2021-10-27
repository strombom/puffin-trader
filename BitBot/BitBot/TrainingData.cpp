#include "pch.h"

#include "TrainingData.h"
#include "BitLib/Logger.h"
#include "BitLib/BitBotConstants.h"

#include <filesystem>


TrainingData::TrainingData()
{
    ground_truth = std::make_shared<std::vector<std::array<int, BitBot::Trading::take_profit.size()>>>();
    ground_truth_timestamps = std::make_shared<std::vector<std::array<time_point_ms, BitBot::Trading::take_profit.size()>>>();
}

void make_section_thread(
    std::string_view symbol,
    std::string_view path,
    const sptrBinanceKlines klines,
    const sptrIndicators indicators,
    time_point_ms timestamp_start,
    time_point_ms timestamp_end,
    std::shared_ptr < std::vector<std::array<int, BitBot::Trading::take_profit.size()>>> ground_truth,
    std::shared_ptr < std::vector<std::array<time_point_ms, BitBot::Trading::take_profit.size()>>> ground_truth_timestamps
) {

    auto prev_ts = indicators->timestamps[0];
    for (auto i = 1; i < indicators->timestamps.size(); i++) {
        const auto new_ts = indicators->timestamps[i];
        if (new_ts == prev_ts) {
            printf("hm %d\n", i);
        }
        prev_ts = new_ts;
    }
    
    std::filesystem::create_directories(path);
    auto file_path = std::string{ path } + "/" + date::format("%F", timestamp_start) + "_" + std::string{ symbol } + ".csv";

    auto csv_file = std::ofstream{ file_path, std::ios::binary };

    csv_file << "\"ind_idx\"";
    csv_file << ",\"timestamp\"";

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
    for (auto idx = 0; idx < BitBot::Trading::take_profit.size(); idx++) {
        csv_file << std::fixed << ",\"(" << BitBot::Trading::take_profit[idx] << "," << BitBot::Trading::stop_loss[idx] << ")\"";
    }

    csv_file << ",\"ground_truth_timestamp\"";
    csv_file << "\n";
    csv_file << std::defaultfloat;

    for (auto gt_idx = 0; gt_idx < ground_truth->size(); gt_idx++) {
        if (indicators->timestamps[gt_idx] < timestamp_start) {
            continue;
        }
        else if (indicators->timestamps[gt_idx] > timestamp_end) {
            break;
        }

        bool skip = false;
        for (auto profit_idx = 0; profit_idx < BitBot::Trading::take_profit.size(); profit_idx++) {
            if ((*ground_truth)[gt_idx][profit_idx] == 0) {
                skip = true;
                break;
            }
        }

        if (!skip) {
            const int indicator_idx = gt_idx;

            csv_file << indicator_idx;
            csv_file << ",\"" << DateTime::to_string_iso_8601(indicators->timestamps[indicator_idx]) << "\"";

            const auto& indicator = indicators->indicators[indicator_idx];
            for (auto i = 0; i < indicator.size(); i++) {
                csv_file << "," << indicator[i];
            }
            csv_file << symbols_string;
            for (auto profit_idx = 0; profit_idx < BitBot::Trading::take_profit.size(); profit_idx++) {
                csv_file << "," << (*ground_truth)[gt_idx][profit_idx];
            }

            csv_file << ",\"" << DateTime::to_string_iso_8601((*ground_truth_timestamps)[gt_idx].back()) << "\"";

            csv_file << "\n";
        }
    }

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
        printf("Warning, loading cached ground truth.\n");
        load_ground_truth(symbol);
    }

    const auto data_length = indicators->indicators.size();
    auto timestamp_train_start = timestamp_start + date::days{ 2 }; // 2 days offset, delay due to 150 intrinsic steps indicator length
    auto train_start_idx = 0;
    
    while (true) {
        const auto timestamp_train_end = timestamp_train_start + BitBot::history_length;
        const auto timestamp_val_start = timestamp_train_end;
        const auto timestamp_val_end = timestamp_val_start + BitBot::validation_length;

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
            const auto gt_max_timestamp = (*ground_truth_timestamps)[indicator_idx].back();
            if (gt_max_timestamp < timestamp_train_end && gt_max_timestamp.time_since_epoch().count() != 0) {
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
            const auto gt_max_timestamp = (*ground_truth_timestamps)[indicator_idx].back();
            if (gt_max_timestamp.time_since_epoch().count() != 0) { // gt_max_timestamp < timestamp_val_end && 
                validation_indices.push_back(indicator_idx);
            }

            indicator_idx++;
            if (indicator_idx == data_length) {
                goto end_of_indicators;
            }
        }

        save_indices(path, "/" + date::format("%F", timestamp_train_start) + "_train_" + std::string{ symbol } + ".csv", training_indices);
        save_indices(path, "/" + date::format("%F", timestamp_val_start) + "_val_" + std::string{ symbol } + ".csv", validation_indices);

        timestamp_train_start += BitBot::validation_length;
    }

end_of_indicators:
    ;
}

void TrainingData::make(std::string_view path, std::string_view symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, klines, indicators);
    //save_ground_truth(symbol);
    //load_ground_truth(symbol);
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

    auto positions = std::array<std::list<Position>, BitBot::Trading::take_profit.size()>{};
    auto ind_idx = 0;

    std::array<int, BitBot::Trading::take_profit.size()> maxcount = { 0 };

    for (auto kline_idx = 0; kline_idx < klines->rows.size(); kline_idx++) {
        const auto mark_price = klines->rows[kline_idx].open;

        // Remove expired positions
        for (auto profit_idx = 0; profit_idx < BitBot::Trading::take_profit.size(); profit_idx++) {
            maxcount[profit_idx] = std::max(maxcount[profit_idx], (int)positions[profit_idx].size());

            auto remove = false;

            auto position = positions[profit_idx].begin();
            while (position != positions[profit_idx].end()) {
                if (mark_price < position->take_profit) {
                    break;
                }
                (*ground_truth)[position->ind_idx][profit_idx] = 1;
                (*ground_truth_timestamps)[position->ind_idx][profit_idx] = klines->rows[kline_idx].open_time; // indicators->timestamps[ind_idx];
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
                    (*ground_truth_timestamps)[position->ind_idx][profit_idx] = klines->rows[kline_idx].open_time; //indicators->timestamps[ind_idx];
                    position->remove = true;
                    remove = true;
                }
            }

            if (remove) {
                positions[profit_idx].remove_if([](const Position& position) { return position.remove; });
            }
        }

        // Add positions
        while (ind_idx < indicators->timestamps.size() && indicators->timestamps[ind_idx] <= klines->rows[kline_idx].open_time) {
            /*
            printf(
                "while kline (%d) %s   event (%d) %s\n", 
                kline_idx,
                DateTime::to_string_iso_8601(klines->rows[kline_idx].timestamp).c_str(), 
                ie_idx,
                DateTime::to_string_iso_8601(intrinsic_events->events[ie_idx].timestamp).c_str()
            );
            */
            //if (ind_idx == indicators->timestamps.size() - 1) {
            //    printf("");
            //}

            for (auto profit_idx = 0; profit_idx < BitBot::Trading::take_profit.size(); profit_idx++) {
                const auto take_profit = (float)(mark_price * BitBot::Trading::take_profit[profit_idx]);
                const auto stop_loss = (float)(mark_price * BitBot::Trading::stop_loss[profit_idx]);

                //auto pos_idx = 0;
                auto position = positions[profit_idx].begin();
                for (; position != positions[profit_idx].end(); ++position) {
                    if (position->take_profit > take_profit) {
                        break;
                    }
                }
                positions[profit_idx].insert(position, { ind_idx, take_profit, stop_loss });
            }

            ind_idx++;
        }
    }

    printf("Maxcount %s:", std::string{ symbol }.c_str());
    for (auto idx = 0; idx < BitBot::Trading::take_profit.size(); idx++) {
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
        for (auto col_idx = 0; col_idx < BitBot::Trading::take_profit.size(); col_idx++) {
            const auto gt_value = (*ground_truth)[row_idx][col_idx];
            const auto gt_timestamp = (*ground_truth_timestamps)[row_idx][col_idx];
            data_file.write(reinterpret_cast<const char*>(&gt_value), sizeof(gt_value));
            data_file.write(reinterpret_cast<const char*>(&gt_timestamp), sizeof(gt_timestamp));
        }
    }

    data_file.close();
}

void TrainingData::load_ground_truth(std::string_view symbol)
{
    const auto file_path = std::string{ BitBot::path } + "\\ground_truth\\" + std::string{ symbol } + ".dat";

    const auto row_size = (sizeof(int) + sizeof(time_point_ms)) * BitBot::Trading::take_profit.size();
    const auto gt_length = std::filesystem::file_size(file_path) / row_size;

    auto data_file = std::ifstream{ file_path, std::ios::binary };
    auto intrinsic_event = IntrinsicEvent{};

    ground_truth->resize(gt_length);
    ground_truth_timestamps->resize(gt_length);

    for (auto row_idx = 0; row_idx < gt_length; row_idx++) {
        for (auto col_idx = 0; col_idx < BitBot::Trading::take_profit.size(); col_idx++) {
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
