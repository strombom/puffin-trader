#include "Logger.h"
#include "Database.h"


Database::Database(const std::string &root_path)
{
    std::string attributes_file_path = root_path + "\\attributes.sqlite";
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

bool Database::has_attribute(const std::string& key_a, const std::string& key_b)
{
    return has_attribute(key_a + "_" + key_b);
}

bool Database::has_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c)
{
    return has_attribute(key_a + "_" + key_b + "_" + key_c);
}

time_point_us Database::get_attribute(const std::string& key, const time_point_us& default_date_time)
{
    //date::format("%F %T", std::chrono::system_clock::now()).c_str()

    SQLite::Statement query_insert(*attributes_db, "INSERT OR IGNORE INTO attributes(\"key\", \"value\") VALUES(:key, :value)");
    query_insert.bind(":key", key.c_str());
    query_insert.bind(":value", date::format("%F %T", default_date_time).c_str()); //default_date_time.to_string().c_str());
    query_insert.executeStep();

    SQLite::Statement query_select(*attributes_db, "SELECT value FROM attributes WHERE key = ?");
    query_select.bind(1, key.c_str());
    query_select.executeStep();

    std::istringstream in(query_select.getColumn(0).getString());
    time_point_us tp;
    in >> date::parse("%F %T", tp);
    return tp;
    //return DateTime(query_select.getColumn(0).getString());
}

time_point_us Database::get_attribute(const std::string& key_a, const std::string& key_b, const time_point_us& default_date_time)
{
    return get_attribute(key_a + "_" + key_b, default_date_time);
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

void Database::set_attribute(const std::string& key_a, const std::string& key_b, const time_point_us& date_time)
{
    set_attribute(key_a + "_" + key_b, date_time);
}

void Database::set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& date_time)
{
    set_attribute(key_a + "_" + key_b + "_" + key_c, date_time);
}

void Database::tick_data_extend(const std::string& exchange, const std::string& symbol, std::shared_ptr<DatabaseTicks> ticks)
{
    time_point_us first_timestamp = ticks->get_first_timestamp();

    //bool has = has_attribute(exchange, "");
}

DatabaseTickRow::DatabaseTickRow(time_point_us timestamp, float price, float volume, bool buy) :
    timestamp(timestamp), price(price), volume(volume), buy(buy)
{

}

void DatabaseTicks::append(time_point_us timestamp, float price, float volume, bool buy)
{
    ticks.push_back(DatabaseTickRow(timestamp, price, volume, buy));
}

time_point_us DatabaseTicks::get_first_timestamp(void)
{
    return ticks[0].timestamp;
}
