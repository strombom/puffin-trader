#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include "SQLiteCpp/SQLiteCpp.h"

#include "DateTime.h"


struct DatabaseTickRow
{
    DatabaseTickRow(time_point_us timestamp, float price, float volume, bool buy);

    time_point_us timestamp;
    float price;
    float volume;
    bool buy;
};

class DatabaseTicks
{
public:
    void append(time_point_us timestamp, float price, float volume, bool buy);

    time_point_us get_first_timestamp(void);

private:
    std::vector<DatabaseTickRow> ticks;
};

class Database
{
public:
    Database(const std::string& root_path);

    static std::shared_ptr<Database> create(const std::string& root_path);

    //bool has_attribute(const std::string& key_a, const std::string& key_b);

    bool has_attribute(const std::string& key);
    bool has_attribute(const std::string& key_a, const std::string& key_b);
    bool has_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c);

    time_point_us get_attribute(const std::string& key,   const time_point_us& default_date_time);
    time_point_us get_attribute(const std::string& key_a, const std::string& key_b, const time_point_us& default_date_time);
    time_point_us get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& default_date_time);

    void set_attribute(const std::string& key,   const time_point_us& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const time_point_us& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const time_point_us& date_time);

    void tick_data_extend(const std::string& exchange, const std::string& symbol, std::shared_ptr<DatabaseTicks> ticks);
    //void append_10s(const std::string& symbol, )

private:
    SQLite::Database *attributes_db;

};

using sptrDatabase = std::shared_ptr<Database>;
