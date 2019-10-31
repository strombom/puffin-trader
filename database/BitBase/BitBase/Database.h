#pragma once

#include <string>
#include "DateTime.h"

class Database
{
public:
    Database(const std::string& root_path);

    bool has_attribute(const std::string& key_a, const std::string& key_b);

    DateTime get_attribute_date(const std::string& key_a, const std::string& key_b);
    void set_attribute_date(const std::string& key_a, const std::string& key_b, const DateTime& date_time);

private:

};
