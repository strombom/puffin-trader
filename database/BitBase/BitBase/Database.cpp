#include "Logger.h"
#include "Database.h"

#include <filesystem>
#include <fstream>


Database::Database(const std::string& _root_path) : root_path(_root_path)
{
    std::string attributes_file_path = _root_path + "\\attributes.sqlite";
    attributes_db = new SQLite::Database(attributes_file_path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

    attributes_db->exec("CREATE TABLE IF NOT EXISTS attributes (key TEXT PRIMARY KEY, value TEXT)");

    /*
    if (!has_attribute("bitmex", "BTCUSD")) {
        set_attribute_date("bitmex", "BTCUSD", DateTime(2019, 10, 31, 20, 42, 10.532));
    }
    */
}

std::shared_ptr<Database> Database::create(const std::string& root_path)
{
    return std::make_shared<Database>(root_path);
}

/*
bool Database::has_attribute(const std::string& key_a, const std::string& key_b)
{
    SQLite::Statement query(*attributes_db, "SELECT COUNT(1) FROM attributes WHERE key = ?");
    query.bind(1, (key_a + "_" + key_b).c_str());
    query.executeStep();
    const int count = query.getColumn(0).getInt();
    if (count > 0) {
        return true;
    } else {
        return false;
    }
}
*/

bool Database::has_attribute(const std::string& key)
{
    SQLite::Statement query_select(*attributes_db, "SELECT EXISTS(SELECT * FROM attributes WHERE key = ?)");
    query_select.bind(1, key.c_str());
    query_select.executeStep();

    auto a = query_select.getColumn(0);

    return true;
}

bool Database::has_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c)
{
    return has_attribute(key_a + "_" + key_b + "_" + key_c);
}

time_point_us Database::get_attribute(const std::string& key, const time_point_us& default_date_time)
{
    SQLite::Statement query_insert(*attributes_db, "INSERT OR IGNORE INTO attributes(\"key\", \"value\") VALUES(:key, :value)");
    query_insert.bind(":key", key.c_str());
    query_insert.bind(":value", date::format("%F %T", default_date_time).c_str());
    query_insert.executeStep();

    SQLite::Statement query_select(*attributes_db, "SELECT value FROM attributes WHERE key = ?");
    query_select.bind(1, key.c_str());
    query_select.executeStep();

    std::istringstream in(query_select.getColumn(0).getString());
    time_point_us tp;
    in >> date::parse("%F %T", tp);
    return tp;
}

time_point_us Database::get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& default_date_time)
{
    return get_attribute(key_a + "_" + key_b + "_" + key_c, default_date_time);
}

void Database::set_attribute(const std::string& key, const time_point_us& date_time)
{
    SQLite::Statement query(*attributes_db, "INSERT OR REPLACE INTO attributes(\"key\", \"value\") VALUES(:key, :value)");
    query.bind(":key",   key.c_str());
    query.bind(":value", date::format("%F %T", date_time).c_str());
    query.exec();
}

void Database::set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& date_time)
{
    set_attribute(key_a + "_" + key_b + "_" + key_c, date_time);
}

void Database::tick_data_extend(const std::string& exchange, const std::string& symbol, std::shared_ptr<DatabaseTicks> ticks, const time_point_us& first_timestamp)
{
    time_point_us last_timestamp = get_attribute(exchange, symbol, "tick_data_last_timestamp", first_timestamp);

    std::filesystem::create_directories(root_path + "/" + exchange);
    std::ofstream file(root_path + "/" + exchange + "/" + symbol + ".dat", std::ofstream::app);
    
    bool in_range = false;
    for (auto&& row : ticks->rows) {

        if (!in_range && (row.timestamp > last_timestamp)) {
            in_range = true;
        }
        if (in_range) {
            file << date::format("%FD%T", row.timestamp).c_str();
            file << "," << row.price << "," << row.volume << ",";
            if (row.buy) {
                file << "BUY";
            }
            else {
                file << "SELL";
            }
            file << "\n";

            last_timestamp = row.timestamp;
        }
    }

    file.close();

    set_attribute(exchange, symbol, "tick_data_last_timestamp", last_timestamp);
}

DatabaseTickRow::DatabaseTickRow(time_point_us timestamp, float price, float volume, bool buy) :
    timestamp(timestamp), price(price), volume(volume), buy(buy)
{

}

void DatabaseTicks::append(time_point_us timestamp, float price, float volume, bool buy)
{
    rows.push_back(DatabaseTickRow(timestamp, price, volume, buy));
}

time_point_us DatabaseTicks::get_first_timestamp(void)
{
    return rows[0].timestamp;
}
