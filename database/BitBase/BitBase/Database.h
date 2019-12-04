#pragma once

#include "DateTime.h"
#include "DatabaseTicks.h"

#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>
#include "SQLiteCpp/SQLiteCpp.h"


class Database
{
public:
    Database(const std::string& _root_path);

    static std::shared_ptr<Database> create(const std::string& root_path);

    const std::string get_attribute(const std::string& key, const std::string& default_string);
    const int get_attribute(const std::string& key, int default_value);
    const time_point_us get_attribute(const std::string& key, const time_point_us& default_date_time);
    const std::vector<std::string> get_attribute(const std::string& key, const std::vector<std::string>& default_string_vector);
    const std::unordered_set<std::string> get_attribute(const std::string& key, const std::unordered_set<std::string>& default_string_set);

    void set_attribute(const std::string& key, const std::string& string);
    void set_attribute(const std::string& key, int value);
    void set_attribute(const std::string& key, const time_point_us& date_time);
    void set_attribute(const std::string& key, const std::vector<std::string>& string_vector);
    void set_attribute(const std::string& key, const std::unordered_set<std::string>& string_set);

    void extend_tick_data(const std::string& exchange, const std::string& symbol, const std::unique_ptr<DatabaseTicks> ticks, const time_point_us& first_timestamp);
    std::unique_ptr<DatabaseTick> get_tick(const std::string& exchange, const std::string& symbol, int row_idx);
    //std::unique_ptr<DatabaseTicks> get_tick_data(int start_row, int row_count);
    //void append_10s(const std::string& symbol, )

    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b,                              const T& default_value) { return get_attribute(key_a + "_" + key_b, default_value); }
    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c,    const T& default_value) { return get_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b,                           const T& default_value) { set_attribute(key_a + "_" + key_b, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& default_value) { set_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }

private:
    SQLite::Database *attributes_db;
    const std::string root_path;

    std::mutex sqlite_mutex;
    std::mutex filedb_mutex;

    static constexpr auto time_format = "%F %T";

};

using sptrDatabase = std::shared_ptr<Database>;

