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
    auto slock = std::scoped_lock{ sqlite_mutex };

    auto query_insert = SQLite::Statement{ *attributes_db, "INSERT OR IGNORE INTO attributes(\"key\", \"value\") VALUES(:key, :value)" };
    query_insert.bind(":key", key);
    query_insert.bind(":value", default_string);
    query_insert.executeStep();

    auto query_select = SQLite::Statement{ *attributes_db, "SELECT value FROM attributes WHERE key = :key" };
    query_select.bind(":key", key);
    query_select.executeStep();

    return query_select.getColumn(0).getString();
}

const int Database::get_attribute(const std::string& key, int default_value)
{
    const auto attribute = get_attribute(key, std::to_string(default_value));
    auto value = std::istringstream{ attribute };
    auto return_value = int{ 0 };
    value >> return_value;
    return return_value;
}

const time_point_us Database::get_attribute(const std::string& key, const time_point_us& default_date_time)
{
    const auto attribute = get_attribute(key, date::format(BitBase::Database::time_format, default_date_time));
    auto value = std::istringstream{ attribute };
    auto time_point = time_point_us{};
    value >> date::parse(BitBase::Database::time_format, time_point);
    return time_point;
}

const std::vector<std::string> Database::get_attribute(const std::string& key, const std::vector<std::string>& default_string_vector)
{
    auto space_separated_string = std::ostringstream{};
    std::copy(default_string_vector.begin(), default_string_vector.end(), std::ostream_iterator<std::string>(space_separated_string, ","));
    auto value_stream = std::istringstream{ get_attribute(key, space_separated_string.str()) };
    auto string_vector = std::vector<std::string>(std::istream_iterator<std::string>{value_stream}, std::istream_iterator<std::string>{});
    return string_vector;
}

const std::unordered_set<std::string> Database::get_attribute(const std::string& key, const std::unordered_set<std::string>& default_string_set)
{
    auto space_separated_string = std::ostringstream{};
    std::copy(default_string_set.begin(), default_string_set.end(), std::ostream_iterator<std::string>(space_separated_string, ","));
    auto value_stream = std::istringstream{ get_attribute(key, space_separated_string.str()) };
    auto string_set = std::unordered_set<std::string>(std::istream_iterator<std::string>{value_stream}, std::istream_iterator<std::string>{});
    return string_set;
}

void Database::set_attribute(const std::string& key, const std::string& string)
{
    auto slock = std::scoped_lock{ sqlite_mutex };

    auto query = SQLite::Statement{ *attributes_db, "INSERT OR REPLACE INTO attributes(\"key\", \"value\") VALUES(:key, :value)" };
    query.bind(":key", key);
    query.bind(":value", string);
    query.exec();
}

void Database::set_attribute(const std::string& key, int value)
{
    set_attribute(key, std::to_string(value));
}

void Database::set_attribute(const std::string& key, const time_point_us& date_time)
{
    set_attribute(key, date::format(BitBase::Database::time_format, date_time));
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

void Database::extend_tick_data(const std::string& exchange, const std::string& symbol, uptrDatabaseTicks ticks, const time_point_us& first_timestamp)
{
    auto slock = std::scoped_lock{ filedb_mutex };

    auto last_timestamp = get_attribute(exchange, symbol, "tick_data_last_timestamp", first_timestamp);

    std::filesystem::create_directories(root_path + "/tick/" + exchange);
    auto file = std::ofstream{ root_path + "/tick/" + exchange + "/" + symbol + ".dat", std::ofstream::app | std::ofstream::binary };
    
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

std::unique_ptr<DatabaseTick> Database::get_tick(const std::string& exchange, const std::string& symbol, int row_idx)
{
    auto slock = std::scoped_lock{ filedb_mutex };
    auto file = std::ifstream{ root_path + "/tick/" + exchange + "/" + symbol + ".dat", std::ifstream::binary };
    file.seekg(DatabaseTick::struct_size * row_idx);
    auto tick = DatabaseTick{};
    file >> tick;
    if (file.bad() || file.fail()) {
        return nullptr;
    }
    else {
        return std::make_unique<DatabaseTick>(tick);
    }
}

void Database::extend_interval_data(const std::string& exchange, const std::string& symbol, const std::string& interval_name, const DatabaseIntervals& intervals_data, const time_point_us& timestamp, int tick_idx)
{
    auto slock = std::scoped_lock{ filedb_mutex };

    std::filesystem::create_directories(root_path + "/interval/" + exchange);
    auto file = std::ofstream{ root_path + "/interval/" + exchange + "/" + symbol + "_" + interval_name + ".dat", std::ofstream::app | std::ofstream::binary };
    file << intervals_data;
    file.close();

    set_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_timestamp", timestamp);
    set_attribute(BitBase::Bitmex::exchange_name, symbol + "_interval_" + interval_name + "_tick_idx", tick_idx);
}
