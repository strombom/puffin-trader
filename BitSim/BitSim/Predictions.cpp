#include "pch.h"
#include "Predictions.h"

#include <filesystem>
#include <fstream>


Predictions::Predictions() {
    
    for (auto& file : std::filesystem::directory_iterator{ "E:/BitBot/predictions" })  //loop through the current folder
    {
        if (file.is_regular_file()) {
            const auto&& symbol = file.path().stem().string();
            printf("Load predictions %s\n", symbol.c_str());
            if (!load_predictions(symbol)) {
                printf("Read %s\n", symbol.c_str());
                read_csv(symbol, file.path());
                save_predictions(symbol);
            }
        }
    }
}

void Predictions::read_csv(const std::string symbol, const std::filesystem::path& path)
{
    data.try_emplace(symbol);
    auto&& entry = data[symbol];

    auto file = std::ifstream{ path.string()};
    auto line = std::string{};
    auto row_idx = int{};
    auto datestring = std::array<char, 26>{};
    
    std::getline(file, line); // Skip header

    while (file >> row_idx) {
        file.ignore(); // Skip comma

        entry.emplace_back();
        auto&& row = entry.back();

        file.read((char*) datestring.data(), 25);
        row.timestamp = DateTime::to_time_point_ms(datestring.data(), "%F %T%z");

        file.ignore(); // Skip comma

        for (auto prediction_idx = 0; prediction_idx < delta_count; prediction_idx++) {
            file >> row.prediction[prediction_idx];
            file.ignore(); // Skip comma
        }

        for (auto ground_truth_idx = 0; ground_truth_idx < delta_count; ground_truth_idx++) {
            file >> row.ground_truth[ground_truth_idx];
            file.ignore(); // Skip comma
        }

        while (file.peek() == '\n' || file.peek() == '\r') {
            file.ignore();
        }
    }
}

void Predictions::save_predictions(const std::string symbol)
{
    auto file_path = std::string{ "E:/BitBot/predictions/cache" };
    std::filesystem::create_directories(file_path);
    file_path += "\\" + symbol + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };
    auto&& entry = data[symbol];

    for (auto row_idx = 0; row_idx < entry.size(); row_idx++) {
        const auto timestamp = entry[row_idx].timestamp.time_since_epoch().count();
        data_file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

        for (auto prediction_idx = 0; prediction_idx < delta_count; prediction_idx++) {
            const auto prediction = entry[row_idx].prediction[prediction_idx];
            data_file.write(reinterpret_cast<const char*>(&prediction), sizeof(prediction));
        }
        for (auto ground_truth_idx = 0; ground_truth_idx < delta_count; ground_truth_idx++) {
            const auto ground_truth = entry[row_idx].ground_truth[ground_truth_idx];
            data_file.write(reinterpret_cast<const char*>(&ground_truth), sizeof(ground_truth));
        }
    }

    data_file.close();
}

bool Predictions::load_predictions(const std::string symbol)
{
    const auto file_path = std::string{ "E:/BitBot/predictions/cache/" } + symbol + ".dat";

    if (!std::filesystem::exists(file_path)) {
        return false;
    }

    const auto row_size = (sizeof(time_point_ms) + delta_count * (sizeof(int) + sizeof(float)));
    const auto row_count = std::filesystem::file_size(file_path) / row_size;

    auto data_file = std::ifstream{ file_path, std::ios::binary };

    data.try_emplace(symbol);
    auto&& entry = data[symbol];
    entry.resize(row_count);

    for (auto row_idx = 0; row_idx < row_count; row_idx++) {
        data_file.read(reinterpret_cast <char*> (&entry[row_idx].timestamp), sizeof(time_point_ms));
        for (auto prediction_idx = 0; prediction_idx < delta_count; prediction_idx++) {
            data_file.read(reinterpret_cast <char*> (&entry[row_idx].prediction[prediction_idx]), sizeof(float));
        }
        for (auto ground_truth_idx = 0; ground_truth_idx < delta_count; ground_truth_idx++) {
            data_file.read(reinterpret_cast <char*> (&entry[row_idx].ground_truth[ground_truth_idx]), sizeof(int));
        }
    }
    data_file.close();

    return true;
}
