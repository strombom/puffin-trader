#pragma once

#include <stdio.h>
#include <string>
#include <vector>
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

    time_point_us get_attribute(const std::string& key, const time_point_us& default_date_time);

    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const T& default_date_time);
    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& default_date_time);

    //std::vector<std::string> get_attribute(const std::string& key, const time_point_us& default_date_time);
    //std::vector<std::string> get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& default_date_time);

    void set_attribute(const std::string& key,   const time_point_us& date_time);

    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const T& date_time);
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& date_time);

    void tick_data_extend(const std::string& exchange, const std::string& symbol, const std::unique_ptr<DatabaseTicks> ticks, const time_point_us& first_timestamp);
    //void append_10s(const std::string& symbol, )

private:
    SQLite::Database *attributes_db;
    const std::string root_path;

};

using sptrDatabase = std::shared_ptr<Database>;
