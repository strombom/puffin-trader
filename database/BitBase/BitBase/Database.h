#pragma once

#include <stdio.h>
#include <string>
#include "SQLiteCpp/SQLiteCpp.h"

#include "DateTime.h"


class Database
{
public:
    Database(const std::string& root_path);

    //bool has_attribute(const std::string& key_a, const std::string& key_b);

    DateTime get_attribute(const std::string& key,   const DateTime& default_date_time);
    DateTime get_attribute(const std::string& key_a, const std::string& key_b, const DateTime& default_date_time);
    DateTime get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& default_date_time);

    void set_attribute(const std::string& key,   const DateTime& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const DateTime& date_time);
    void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const DateTime& date_time);

private:
    SQLite::Database *attributes_db;

};
