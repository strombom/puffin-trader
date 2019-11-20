#pragma once

#include <stdio.h>
#include <string>
#include "SQLiteCpp/SQLiteCpp.h"

#include "DateTime.h"


class DatabaseTicks
{
public:
    void append(const DateTime& timestamp, float price, float volume, bool buy);

private:
    std::vector<DateTime> timestamps;
    std::vector<float> prices;
    std::vector<float> volumes;
    std::vector<bool> buys;
};


class Database
{
public:
    Database(const std::string& root_path);

    static std::shared_ptr<Database> create(const std::string& root_path);

    //bool has_attribute(const std::string& key_a, const std::string& key_b);

    DateTime get_attribute(const std::string& key,   const DateTime& default_date_time);
    DateTime get_attribute(const std::string& key_a, const std::string& key_b, const DateTime& default_date_time);
    DateTime get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& default_date_time);

    void set_attribute(const std::string& key,   const DateTime& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const DateTime& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& date_time);

    //void append_10s(const std::string& symbol, )

private:
    SQLite::Database *attributes_db;

};

using sptrDatabase = std::shared_ptr<Database>;
