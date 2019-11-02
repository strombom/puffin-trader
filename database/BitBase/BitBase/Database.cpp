#include "Logger.h"
#include "Database.h"


Database::Database(const std::string &root_path)
{
    std::string attributes_file_path = root_path + "\\attributes.sqlite";
    attributes_db = new SQLite::Database(attributes_file_path, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

    attributes_db->exec("CREATE TABLE IF NOT EXISTS attributes (key TEXT PRIMARY KEY, value TEXT)");

    if (!has_attribute("bitmex", "BTCUSD")) {
        set_attribute_date("bitmex", "BTCUSD", DateTime(2019, 10, 31, 20, 42, 10.532));
    }
}

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

DateTime Database::get_attribute_date(const std::string& key_a, const std::string& key_b)
{
    SQLite::Statement query(*attributes_db, "SELECT COUNT(1) FROM attributes WHERE key = ?");
    query.bind(1, (key_a + "_" + key_b).c_str());
    query.executeStep();
    return DateTime();
}

void Database::set_attribute_date(const std::string& key_a, const std::string& key_b, const DateTime& date_time)
{
    SQLite::Statement query(*attributes_db, "INSERT OR REPLACE INTO attributes(\"key\", \"value\") VALUES(?, ?)");
    query.bind(1, (key_a + "_"  + key_b).c_str());
    query.bind(2, date_time.to_string().c_str());
    query.exec();
}
