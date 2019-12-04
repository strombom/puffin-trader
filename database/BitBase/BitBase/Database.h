#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_set>
#include "SQLiteCpp/SQLiteCpp.h"

#include "DateTime.h"


struct DatabaseTickRow
{
    DatabaseTickRow(const time_point_us timestamp, const float price, const float volume, const bool buy) :
        timestamp(timestamp), price(price), volume(volume), buy(buy) {}

    friend std::ostream& operator<<(std::ostream& stream, const DatabaseTickRow& row);
    friend std::istream& operator>>(std::istream& stream, DatabaseTickRow& row);

    const time_point_us timestamp;
    const float price;
    const float volume;
    const bool buy;
};

class DatabaseTicks
{
public:
    DatabaseTicks(void);

    void append(const time_point_us timestamp, const float price, const float volume, const bool buy);

    time_point_us get_first_timestamp(void);

    friend std::istream& operator>>(std::istream& stream, DatabaseTicks& row);

    std::vector<DatabaseTickRow> rows;
};

class Database
{
public:
    Database(const std::string& _root_path);

    static std::shared_ptr<Database> create(const std::string& root_path);

    bool has_attribute(const std::string& key);
    bool has_attribute(const std::string& key_a, const std::string& key_b);
    bool has_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c);

    const std::string get_attribute(const std::string& key, const std::string& default_string);
    time_point_us get_attribute(const std::string& key, const time_point_us& default_date_time);
    std::vector<std::string> get_attribute(const std::string& key, const std::vector<std::string>& default_string_vector);
    std::unordered_set<std::string> get_attribute(const std::string& key, const std::unordered_set<std::string>& default_string_set);

    void set_attribute(const std::string& key, const std::string& string);
    void set_attribute(const std::string& key, const time_point_us& date_time);
    void set_attribute(const std::string& key, const std::vector<std::string>& string_vector);
    void set_attribute(const std::string& key, const std::unordered_set<std::string>& string_set);

    void tick_data_extend(const std::string& exchange, const std::string& symbol, const std::unique_ptr<DatabaseTicks> ticks, const time_point_us& first_timestamp);
    //void append_10s(const std::string& symbol, )

    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b,                              const T& default_value) { return get_attribute(key_a + "_" + key_b, default_value); }
    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c,    const T& default_value) { return get_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b,                           const T& default_value) { set_attribute(key_a + "_" + key_b, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& default_value) { set_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }

private:
    SQLite::Database *attributes_db;
    const std::string root_path;

    static constexpr auto time_format = "%F %T";

};

using sptrDatabase = std::shared_ptr<Database>;

