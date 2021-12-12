#include "TickData.h"
#include "Config.h"


TickData::TickData(const Symbol& symbol)
{
    auto count = 0;
    auto start = 0, end = 0;
    for (auto const& file_path : std::filesystem::directory_iterator{ std::string{ Config::base_path } + "csv/" })
    {
        auto csv_reader = io::CSVReader<3>{ file_path.path().string() };
        csv_reader.read_header(io::ignore_extra_column, "timestamp", "price", "size");
        double timestamp_s;
        float price;
        float size;
        while (csv_reader.read_row(timestamp_s, price, size)) {
            const auto timestamp = std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>{ std::chrono::microseconds{ (unsigned long long) (timestamp_s * 1000000) } };
            rows.push_back({ timestamp, price, size });
            end++;
        }

        std::reverse(rows.begin() + start, rows.begin() + end);

        count++;
        if (count == 3) {
            break;
        }
        start = end;
    }
}

void TickData::save_csv(std::string file_path)
{
    auto csv_file = std::ofstream{ file_path, std::ios::binary };
    csv_file << "\"timestamp\",\"price\",\"size\"\n";
    csv_file << std::fixed;
    for (const auto& row : rows) {
        const auto timestamp = std::get<0>(row).time_since_epoch().count() / 1000000.0;
        csv_file.precision(6);
        csv_file << timestamp << ",";
        csv_file.precision(2);
        csv_file << std::get<1>(row) << ",";
        csv_file.precision(3);
        csv_file << std::get<2>(row) << "\n";
    }
    csv_file.close();
}