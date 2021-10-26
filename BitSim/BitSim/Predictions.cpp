#include "pch.h"
#include "Predictions.h"

#include <filesystem>
#include <fstream>


Predictions::Predictions(void)
{
    for (const auto& symbol : symbols) {
        printf("Load predictions %s\n", symbol.name.data());
        if (!load(symbol)) {
            printf("Read csv %s\n", symbol.name.data());
            load_csv(symbol);
            save(symbol);
        }
    }
}

void Predictions::load_csv(const Symbol& symbol)
{
    data[symbol.idx].clear();
    data_idx[symbol.idx] = 0;

    auto&& entry = data[symbol.idx];

    const auto file_path = std::string{ BitSim::Predictions::path } + symbol.name.data() + ".csv";
    auto file = std::ifstream{ file_path };
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

        for (auto prediction_idx = 0; prediction_idx < BitBot::TrainingData::delta_count; prediction_idx++) {
            file >> row.prediction[prediction_idx];
            file.ignore(); // Skip comma
        }

        for (auto ground_truth_idx = 0; ground_truth_idx < BitBot::TrainingData::delta_count; ground_truth_idx++) {
            file >> row.ground_truth[ground_truth_idx];
            file.ignore(); // Skip comma
        }

        while (file.peek() == '\n' || file.peek() == '\r') {
            file.ignore();
        }
    }
}

void Predictions::save(const Symbol& symbol)
{
    auto file_path = std::string{ BitSim::Predictions::path } + "cache/";
    std::filesystem::create_directories(file_path);
    file_path += std::string{ symbol.name.data() } + ".dat";

    auto data_file = std::ofstream{ file_path, std::ios::binary };
    auto&& entry = data[symbol.idx];

    for (auto row_idx = 0; row_idx < entry.size(); row_idx++) {
        const auto timestamp = entry[row_idx].timestamp.time_since_epoch().count();
        data_file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

        for (auto prediction_idx = 0; prediction_idx < BitBot::TrainingData::delta_count; prediction_idx++) {
            const auto prediction = entry[row_idx].prediction[prediction_idx];
            data_file.write(reinterpret_cast<const char*>(&prediction), sizeof(prediction));
        }
        for (auto ground_truth_idx = 0; ground_truth_idx < BitBot::TrainingData::delta_count; ground_truth_idx++) {
            const auto ground_truth = entry[row_idx].ground_truth[ground_truth_idx];
            data_file.write(reinterpret_cast<const char*>(&ground_truth), sizeof(ground_truth));
        }
    }

    data_file.close();
}

bool Predictions::load(const Symbol& symbol)
{
    const auto file_path = std::string{ BitSim::Predictions::path } + "cache/" + symbol.name.data() + ".dat";

    if (!std::filesystem::exists(file_path)) {
        return false;
    }

    const auto row_size = (sizeof(time_point_ms) + BitBot::TrainingData::delta_count * (sizeof(int) + sizeof(float)));
    const auto row_count = std::filesystem::file_size(file_path) / row_size;

    auto data_file = std::ifstream{ file_path, std::ios::binary };

    data[symbol.idx].clear();
    data_idx[symbol.idx] = 0;

    auto&& entry = data[symbol.idx];
    entry.resize(row_count);

    for (auto row_idx = 0; row_idx < row_count; row_idx++) {
        data_file.read(reinterpret_cast <char*> (&entry[row_idx].timestamp), sizeof(time_point_ms));
        for (auto prediction_idx = 0; prediction_idx < BitBot::TrainingData::delta_count; prediction_idx++) {
            data_file.read(reinterpret_cast <char*> (&entry[row_idx].prediction[prediction_idx]), sizeof(float));
        }
        for (auto ground_truth_idx = 0; ground_truth_idx < BitBot::TrainingData::delta_count; ground_truth_idx++) {
            data_file.read(reinterpret_cast <char*> (&entry[row_idx].ground_truth[ground_truth_idx]), sizeof(int));
        }
    }
    data_file.close();

    return true;
}

void Predictions::step_idx(time_point_ms timestamp)
{
    for (const auto& symbol : symbols) {
        while (data[symbol.idx][data_idx[symbol.idx]].timestamp < timestamp && data_idx[symbol.idx] + 1 < data[symbol.idx].size()) {
            data_idx[symbol.idx]++;
        }
        if (data[symbol.idx][data_idx[symbol.idx]].timestamp == timestamp) {
            active[symbol.idx] = true;
        }
        else {
            active[symbol.idx] = false;
        }
    }
}

bool Predictions::has_prediction(const Symbol& symbol)
{
    return active[symbol.idx];
}

double Predictions::get_prediction_score(const Symbol& symbol, int delta_idx)
{
    return data[symbol.idx][data_idx[symbol.idx]].prediction[delta_idx];
}
