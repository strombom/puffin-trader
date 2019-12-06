#pragma once

#include "DateTime.h"
#include "DatabaseTicks.h"
#include "DatabaseIntervals.h"

#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>
#include "SQLiteCpp/SQLiteCpp.h"


class TickTableRead
{
public:
    TickTableRead(const std::string& root_path, const std::string& exchange, const std::string& symbol);
    ~TickTableRead(void);

    std::unique_ptr<DatabaseTick> get_tick(int tick_idx);
    std::unique_ptr<DatabaseTick> get_next_tick(void);

private:
    const std::string& root_path;
    std::mutex file_mutex;
    std::ifstream file;

    std::unique_ptr<DatabaseTick> _get_tick(void);
};

class TickTableWrite
{
public:
    TickTableWrite(const std::string& root_path, const std::string& exchange, const std::string& symbol);
    ~TickTableWrite(void);

    void extend(uptrDatabaseTicks ticks, const time_point_us& first_timestamp);

private:
    const std::string& root_path;
    std::mutex file_mutex;
    std::ofstream file;
};

class IntervalTable
{
    
public:

};

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

    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b,                              const T& default_value) { return get_attribute(key_a + "_" + key_b, default_value); }
    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c,    const T& default_value) { return get_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b,                           const T& default_value) { set_attribute(key_a + "_" + key_b, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& default_value) { set_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }

    std::unique_ptr<TickTableRead> open_tick_table_read(const std::string& exchange, const std::string& symbol);
    std::unique_ptr<TickTableWrite> open_tick_table_write(const std::string& exchange, const std::string& symbol);

    //void extend_tick_data(const std::string& exchange, const std::string& symbol, uptrDatabaseTicks ticks, const time_point_us& first_timestamp);
    //std::unique_ptr<DatabaseTick> get_tick(const std::string& exchange, const std::string& symbol, int row_idx);

    //std::unique_ptr<IntervalTable> open_interval_table(const std::string& exchange, const std::string& symbol, const std::chrono::seconds& interval);

    void extend_interval_data(const std::string& exchange, const std::string& symbol, const std::string& interval_name, const DatabaseIntervals& intervals_data, const time_point_us& timestamp, int tick_idx);

private:
    std::unique_ptr<SQLite::Database> attributes_db;
    const std::string root_path;

    std::mutex sqlite_mutex;
};

using sptrDatabase = std::shared_ptr<Database>;
