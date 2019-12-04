#include "Logger.h"
#include "Database.h"

#include <filesystem>
#include <fstream>


Database::Database(const std::string& _root_path) : 
    root_path(_root_path)
{
    const auto attributes_file_path = _root_path + "\\attributes.sqlite";
    attributes_db = new SQLite::Database{ attributes_file_path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE };

    attributes_db->exec("CREATE TABLE IF NOT EXISTS attributes (key TEXT PRIMARY KEY, value TEXT)");
}

std::shared_ptr<Database> Database::create(const std::string& root_path)
{
    return std::make_shared<Database>(root_path);
}

const std::string Database::get_attribute(const std::string& key, const std::string& default_string)
{
    auto query_insert = SQLite::Statement{ *attributes_db, "INSERT OR IGNORE INTO attributes(\"key\", \"value\") VALUES(:key, :value)" };
    query_insert.bind(":key", key);
    query_insert.bind(":value", default_string);
    query_insert.executeStep();

    auto query_select = SQLite::Statement{ *attributes_db, "SELECT value FROM attributes WHERE key = :key" };
    query_select.bind(":key", key);
    query_select.executeStep();

    return query_select.getColumn(0).getString();
}

time_point_us Database::get_attribute(const std::string& key, const time_point_us& default_date_time)
{
    const auto attribute = get_attribute(key, date::format("%F %T", default_date_time));
    auto value = std::istringstream{ attribute };
    auto time_point = time_point_us{};
    value >> date::parse("%F %T", time_point);
    return time_point;
}

std::vector<std::string> Database::get_attribute(const std::string& key, const std::vector<std::string>& default_string_vector)
{
    auto space_separated_string = std::ostringstream{};
    std::copy(default_string_vector.begin(), default_string_vector.end(), std::ostream_iterator<std::string>(space_separated_string, ","));
    auto value_stream = std::istringstream{ get_attribute(key, space_separated_string.str()) };
    auto string_vector = std::vector<std::string>(std::istream_iterator<std::string>{value_stream}, std::istream_iterator<std::string>{});
    return string_vector;
}

std::unordered_set<std::string> Database::get_attribute(const std::string& key, const std::unordered_set<std::string>& default_string_set)
{
    auto space_separated_string = std::ostringstream{};
    std::copy(default_string_set.begin(), default_string_set.end(), std::ostream_iterator<std::string>(space_separated_string, ","));
    auto value_stream = std::istringstream{ get_attribute(key, space_separated_string.str()) };
    auto string_set = std::unordered_set<std::string>(std::istream_iterator<std::string>{value_stream}, std::istream_iterator<std::string>{});
    return string_set;
}

void Database::set_attribute(const std::string& key, const std::string& string)
{
    auto query = SQLite::Statement{ *attributes_db, "INSERT OR REPLACE INTO attributes(\"key\", \"value\") VALUES(:key, :value)" };
    query.bind(":key", key);
    query.bind(":value", string);
    query.exec();
}

void Database::set_attribute(const std::string& key, const time_point_us& date_time)
{
    set_attribute(key, date::format("%F %T", date_time));
}

void Database::set_attribute(const std::string& key, const std::vector<std::string>& string_vector)
{
    std::ostringstream space_separated_list;
    std::copy(string_vector.begin(), string_vector.end(), std::ostream_iterator<std::string>(space_separated_list, " "));

    set_attribute(key, space_separated_list.str());
}

void Database::set_attribute(const std::string& key, const std::unordered_set<std::string>& string_vector)
{
    std::ostringstream space_separated_list;
    std::copy(string_vector.begin(), string_vector.end(), std::ostream_iterator<std::string>(space_separated_list, " "));

    set_attribute(key, space_separated_list.str());
}

void Database::tick_data_extend(const std::string& exchange, const std::string& symbol, const std::unique_ptr<DatabaseTicks> ticks, const time_point_us& first_timestamp)
{
    auto last_timestamp = get_attribute(exchange, symbol, "tick_data_last_timestamp", first_timestamp);

    std::filesystem::create_directories(root_path + "/tick/" + exchange);
    auto file = std::ofstream{ root_path + "/tick/" + exchange + "/" + symbol + ".dat", std::ofstream::app };
    
    auto in_range = false;
    for (auto&& row : ticks->rows) {
        if (!in_range && (row.timestamp > last_timestamp)) {
            in_range = true;
        }
        if (in_range) {
            file << row;
            last_timestamp = row.timestamp;
        }
    }

    file.close();

    set_attribute(exchange, symbol, "tick_data_last_timestamp", last_timestamp);
}

std::ostream& operator<<(std::ostream& stream, const DatabaseTickRow& row)
{
    const auto timestamp = row.timestamp.time_since_epoch().count();
    const auto price = row.price;
    const auto volume = row.volume;
    const auto buy = row.buy;

    stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&price), sizeof(price));
    stream.write(reinterpret_cast<const char*>(&volume), sizeof(volume));
    stream.write(reinterpret_cast<const char*>(&buy), sizeof(buy));

    return stream;
}

std::istream& operator>>(std::istream& stream, DatabaseTickRow& row)
{
    /*
    const auto timestamp = row.timestamp.time_since_epoch().count();
    const auto price = row.price;
    const auto volume = row.volume;
    const auto buy = row.buy;

    stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&price), sizeof(price));
    stream.write(reinterpret_cast<const char*>(&volume), sizeof(volume));
    stream.write(reinterpret_cast<const char*>(&buy), sizeof(buy));

    return stream;
    */

    return stream;
}

DatabaseTicks::DatabaseTicks(void)
{

}

std::istream& operator>>(std::istream& stream, DatabaseTicks& row)
{
    /*
    const auto timestamp = row.timestamp.time_since_epoch().count();
    const auto price = row.price;
    const auto volume = row.volume;
    const auto buy = row.buy;

    stream.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
    stream.write(reinterpret_cast<const char*>(&price), sizeof(price));
    stream.write(reinterpret_cast<const char*>(&volume), sizeof(volume));
    stream.write(reinterpret_cast<const char*>(&buy), sizeof(buy));

    return stream;
    */

    return stream;
}

void DatabaseTicks::append(const time_point_us timestamp, const float price, const float volume, const bool buy)
{
    rows.push_back(DatabaseTickRow(timestamp, price, volume, buy));
}

time_point_us DatabaseTicks::get_first_timestamp(void)
{
    return rows[0].timestamp;
}
