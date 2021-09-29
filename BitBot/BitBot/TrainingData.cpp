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
    const std::string& path,
    const sptrBinanceKlines klines,
    const sptrIndicators indicators,
    time_point_ms timestamp_start,
    time_point_ms timestamp_end,
    std::shared_ptr < std::vector<std::array<int, BitBot::TrainingData::take_profit.size()>>> ground_truth,
    std::shared_ptr < std::vector<std::array<time_point_ms, BitBot::TrainingData::take_profit.size()>>> ground_truth_timestamps
) {
    
    std::filesystem::create_directories(path);
    auto file_path = path + "/" + date::format("%F", timestamp_start) + "_" + symbol + ".csv";

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

    for (auto gt_idx = 0; gt_idx < ground_truth->size(); gt_idx++) {
        if (indicators->timestamps[gt_idx] < timestamp_start) {
            continue;
        } else if (indicators->timestamps[gt_idx] > timestamp_end) {
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

        const int indicator_idx = gt_idx - (BitBot::Indicators::max_length - 1);

        csv_file << "\"" << DateTime::to_string_iso_8601(indicators->timestamps.at(indicator_idx)) << "\"";
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

#include <xtensor-io/xnpz.hpp>

void TrainingData::make_sections(const std::string& path, const std::string& symbol, const sptrBinanceKlines klines, const sptrIndicators indicators)
{
    const auto bbs = BitBot::symbols.size();
    auto vec_symbols = std::vector<bool>{};
    auto vec_indicators = std::vector<float>{};
    auto vec_predictions = std::vector<int>{};
    auto vec_timestamps = std::vector<int>{};

    if (false) {
        if (ground_truth_symbol != symbol) {
            for (auto& thread : threads) {
                thread.join();
            }
            threads.clear();
            make_ground_truth(symbol, klines, indicators);
        }
        save_ground_truth(symbol);
    }
    else {
        load_ground_truth(symbol);
    }
    
    auto timestamp_train_start = date::floor<date::days>(indicators->timestamps.at(0));
    auto train_start_idx = 0;

    const auto data_length = indicators->indicators.size();
    const auto profit_idx = BitBot::TrainingData::take_profit.size() - 1;

    while (true) {
        const auto timestamp_train_end = timestamp_train_start + BitBot::history_length;
        const auto timestamp_val_start = timestamp_train_end;
        const auto timestamp_val_end = timestamp_val_start + date::days{ 1 };

        while (indicators->timestamps[train_start_idx] < timestamp_train_start) {
            train_start_idx++;
            if (train_start_idx == data_length) {
                goto end_of_indicators;
            }
        }
        if (indicators->timestamps[train_start_idx].time_since_epoch().count() == 0) {
            goto end_of_indicators;
        }

        auto train_idx = train_start_idx;
        while (ground_truth_timestamps->at(train_idx)[profit_idx] < timestamp_train_end) {

            //ground_truth_timestamps->at(train_idx)[profit_idx] < timestamp_train_end

            train_idx++;
            if (train_idx == data_length) {
                goto end_of_indicators;
            }
        }
        /*
        train_end_idx--;
        if (indicators->timestamps[train_end_idx].time_since_epoch().count() == 0) {
            goto end_of_indicators;
        }
        
        auto val_start_idx = train_end_idx;
        while (indicators->timestamps[val_start_idx] < timestamp_val_start) {
            val_start_idx++;
            if (val_start_idx == data_length) {
                goto end_of_indicators;
            }
        }
        if (indicators->timestamps[val_start_idx].time_since_epoch().count() == 0) {
            goto end_of_indicators;
        }

        auto val_end_idx = val_start_idx;
        while (ground_truth_timestamps->at(val_end_idx)[profit_idx] < timestamp_val_end) {
            val_end_idx++;
            if (val_end_idx == data_length || indicators->timestamps[val_end_idx].time_since_epoch().count() == 0) {
                break;
            }
        }
        val_end_idx--;
        if (val_end_idx - val_start_idx < 10) {
            printf("warning %d - %d = %d:\n", val_end_idx, val_start_idx, val_end_idx - val_start_idx);
            //goto end_of_indicators;
        }
        

        printf("section %s - %s:\n",
            DateTime::to_string_iso_8601(timestamp_train_start).c_str(),
            DateTime::to_string_iso_8601(timestamp_train_end).c_str()
        );
        if (train_start_idx == 0) {

            //printf(" train start %s  %s  %s\n",
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 0)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 1)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 2)[profit_idx]).c_str()
            //);
        }
        else if (train_start_idx == 1) {

            //printf(" train start %s  %s  %s  %s\n",
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx - 1)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 0)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 1)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 2)[profit_idx]).c_str()
            //);
        }
        else {
            //printf(" train start %s  %s  %s  %s  %s\n",
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx - 2)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx - 1)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 0)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 1)[profit_idx]).c_str(),
            //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_start_idx + 2)[profit_idx]).c_str()
            //);
        }
        //printf(" train end   %s  %s  %s  %s  %s\n",
        //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx - 2)[profit_idx]).c_str(),
        //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx - 1)[profit_idx]).c_str(),
        //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx + 0)[profit_idx]).c_str(),
        //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx + 1)[profit_idx]).c_str(),
        //    DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx + 2)[profit_idx]).c_str()
        //);

        
        printf(" val start  %s  %s  %s  %s  %s\n",
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_start_idx - 2)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_start_idx - 1)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_start_idx + 0)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_start_idx + 1)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_start_idx + 2)[profit_idx]).c_str()
        );
        printf(" val end    %s  %s  %s  %s  %s\n",
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx - 2)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx - 1)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx + 0)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx + 1)[profit_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx + 2)[profit_idx]).c_str()
        );
        */
        
        /*
        printf(" train (%6d - %6d) %s - %s (gt: %s)\n",
            train_start_idx,
            train_end_idx,
            DateTime::to_string_iso_8601(indicators->timestamps[train_start_idx]).c_str(),
            DateTime::to_string_iso_8601(indicators->timestamps[train_end_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(train_end_idx)[profit_idx]).c_str()
        );
        printf(" val   (%6d - %6d) %s - %s (gt: %s)\n",
            val_start_idx,
            val_end_idx,
            DateTime::to_string_iso_8601(indicators->timestamps[val_start_idx]).c_str(),
            DateTime::to_string_iso_8601(indicators->timestamps[val_end_idx]).c_str(),
            DateTime::to_string_iso_8601(ground_truth_timestamps->at(val_end_idx)[profit_idx]).c_str()
        );
        */

        timestamp_train_start += date::days{ 1 };
    }

end_of_indicators:
    printf("end_of_indicators\n");



    auto symbols = xt::xtensor_fixed<bool, xt::xshape<2, 1>>{};

    symbols.at(0, 0) = true;
    symbols.at(1, 0) = false;

    auto table = xt::xtensor_fixed<double, xt::xshape<2, 2>>{};

    table.at(0, 0) = 1;
    table.at(0, 1) = 2;
    table.at(1, 0) = 4;
    table.at(1, 1) = 8;

    xt::dump_npz(std::string{ BitBot::path } + "/test.npz", "testvar", table, false, false);

    table.at(0, 0) = 2;
    table.at(0, 1) = 3;
    table.at(1, 0) = 5;
    table.at(1, 1) = 9;

    xt::dump_npz(std::string{ BitBot::path } + "/test.npz", "testvar2", table, false, true);

    xt::dump_npz(std::string{ BitBot::path } + "/test.npz", "symbols", symbols, false, true);


}

/*
void TrainingData::make_section(const std::string& path, const std::string& symbol, const std::string& suffix, const sptrBinanceKlines klines,  const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    if (ground_truth_symbol != symbol) {
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
        make_ground_truth(symbol, klines, indicators);
    }

    make_section_thread(symbol, suffix, path, klines, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps);

    //auto thread = std::thread{ make_section_thread, symbol, suffix, path, klines, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps };
    //threads.push_back(std::move(thread));

    //if (threads.size() == (int) (std::thread::hardware_concurrency() * 1.3)) {
    //    threads.begin()->join();
    //    threads.erase(threads.begin());
    //}
}
*/

void TrainingData::make(const std::string& path, const std::string& symbol, const sptrBinanceKlines klines, const sptrIndicators indicators, time_point_ms timestamp_start, time_point_ms timestamp_end)
{
    make_ground_truth(symbol, klines, indicators);
    make_section_thread(symbol, path, klines, indicators, timestamp_start, timestamp_end, ground_truth, ground_truth_timestamps);
}
 
void TrainingData::make_ground_truth(const std::string symbol, const sptrBinanceKlines klines, const sptrIndicators indicators)
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

    printf("Maxcount %s:", symbol.c_str());
    for (auto idx = 0; idx < BitBot::TrainingData::take_profit.size(); idx++) {
        printf("   (%d) %d", idx, maxcount[idx]);
    }
    printf("\n");
}

void TrainingData::save_ground_truth(const std::string& symbol)
{
    auto file_path = std::string{ BitBot::path } + "\\ground_truth";
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

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

void TrainingData::load_ground_truth(const std::string& symbol)
{
    const auto file_path = std::string{ BitBot::path } + "\\ground_truth\\" + symbol + ".dat";

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
