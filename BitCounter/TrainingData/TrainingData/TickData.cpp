#include "TickData.h"
#include "Config.h"


TickData::TickData(const Symbol& symbol) : symbol(symbol)
{
    for (auto const& file_path : std::filesystem::directory_iterator{ Config::csv_path })
    {
        auto csv_reader = io::CSVReader<2>{ file_path.path().string() };
        csv_reader.read_header(io::ignore_extra_column, "timestamp", "price");
        double timestamp_s;
        float price;
        while (csv_reader.read_row(timestamp_s, price)) {
            auto timestamp = std::chrono::microseconds{ (unsigned long long) (timestamp_s * 1000000) };
            timestamps.push_back(std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds>{timestamp});
            prices.push_back(price);
        }
        break;
    }
}
