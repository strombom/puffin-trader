#include "BitmexDaily.h"

#pragma warning (disable : 4251)

#include "boost/filesystem.hpp"

#include <fstream>

#pragma pack(1)
struct TradeRow
{
    std::int64_t timestamp;
    float        price;
    float        volume;
    bool         buy;
};

BitmexDaily::BitmexDaily(const std::string& _symbol)
{
    symbol = _symbol;

    std::string root_dir("C:\\development\\github\\puffin-trader\\database\\data");
    std::string data_dir = root_dir + "\\bitmex";
    std::string file_path = data_dir + "\\" + symbol + ".dat";

    // Create directory if not exists
    if (!boost::filesystem::exists(root_dir)) boost::filesystem::create_directory(root_dir);
    if (!boost::filesystem::exists(data_dir)) boost::filesystem::create_directory(data_dir);

    std::ofstream file(file_path, std::ofstream::binary | std::ofstream::app);

    TradeRow tr;
    tr.timestamp = 1;
    tr.price = 2.2f;
    tr.volume = 3.3f;
    tr.buy = true;

    file.write(reinterpret_cast<char*>(&tr), sizeof(tr));

    file.close();
}
