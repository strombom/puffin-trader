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

DateTime Database::get_attribute(const std::string& key, const DateTime& default_date_time)
{
    SQLite::Statement query_insert(*attributes_db, "INSERT OR IGNORE INTO attributes(\"key\", \"value\") VALUES(:key, :value)");
    query_insert.bind(":key", key.c_str());
    query_insert.bind(":value", default_date_time.to_string().c_str());
    query_insert.executeStep();

    SQLite::Statement query_select(*attributes_db, "SELECT value FROM attributes WHERE key = ?");
    query_select.bind(1, key.c_str());
    query_select.executeStep();

    return DateTime(query_select.getColumn(0).getString());
}

DateTime Database::get_attribute(const std::string& key_a, const std::string& key_b, const DateTime& default_date_time)
{
    return get_attribute(key_a + "_" + key_b, default_date_time);
}

DateTime Database::get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& default_date_time)
{
    return get_attribute(key_a + "_" + key_b + "_" + key_c, default_date_time);
}

void Database::set_attribute(const std::string& key, const DateTime& date_time)
{
    SQLite::Statement query(*attributes_db, "INSERT OR REPLACE INTO attributes(\"key\", \"value\") VALUES(:key, :value)");
    query.bind(":key",   key.c_str());
    query.bind(":value", date_time.to_string().c_str());
    query.exec();
}

void Database::set_attribute(const std::string& key_a, const std::string& key_b, const DateTime& date_time)
{
    set_attribute(key_a + "_" + key_b, date_time);
}

void Database::set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& date_time)
{
    set_attribute(key_a + "_" + key_b + "_" + key_c, date_time);
}
