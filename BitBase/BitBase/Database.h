#pragma once
#include "pch.h"

#include "Ticks.h"
#include "DateTime.h"
#include "Intervals.h"

#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_set>


class TickTableRead
{
public:
    TickTableRead(const std::string& root_path, const std::string& exchange, const std::string& symbol);
    ~TickTableRead(void);

    std::unique_ptr<Tick> get_tick(int tick_idx);
    std::unique_ptr<Tick> get_next_tick(void);

private:
    const std::string& root_path;
    std::mutex file_mutex;
    std::ifstream file;

    std::unique_ptr<Tick> _get_tick(void);
};

class Database
{
public:
    Database(const std::string& _root_path);

    static std::shared_ptr<Database> create(const std::string& root_path);

    const std::string get_attribute(const std::string& key, const std::string& default_string);
    const int get_attribute(const std::string& key, int default_value);
    const time_point_ms get_attribute(const std::string& key, const time_point_ms& default_date_time);
    const std::vector<std::string> get_attribute(const std::string& key, const std::vector<std::string>& default_string_vector);
    const std::unordered_set<std::string> get_attribute(const std::string& key, const std::unordered_set<std::string>& default_string_set);

    void set_attribute(const std::string& key, const std::string& string);
    void set_attribute(const std::string& key, int value);
    void set_attribute(const std::string& key, const time_point_ms& date_time);
    void set_attribute(const std::string& key, const std::vector<std::string>& string_vector);
    void set_attribute(const std::string& key, const std::unordered_set<std::string>& string_set);

    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b,                              const T& default_value) { return get_attribute(key_a + "_" + key_b, default_value); }
    template<class T> T get_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c,    const T& default_value) { return get_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b,                           const T& default_value) { set_attribute(key_a + "_" + key_b, default_value); }
    template<class T> void set_attribute(const std::string& key_a, const std::string& key_b, const std::string& key_c, const T& default_value) { set_attribute(key_a + "_" + key_b + "_" + key_c, default_value); }

    std::unique_ptr<TickTableRead> open_tick_table_read(const std::string& exchange, const std::string& symbol);

    void extend_tick_data(const std::string& exchange, const std::string& symbol, uptrDatabaseTicks ticks, const time_point_ms& first_timestamp);
    void extend_interval_data(const std::string& exchange, const std::string& symbol, const std::chrono::milliseconds interval, const Intervals& intervals_data, const time_point_ms& timestamp, int tick_idx);

    std::unique_ptr<Intervals> get_intervals(const std::string& exchange, const std::string& symbol, const time_point_ms& timestamp_start, const time_point_ms& timestamp_end, const std::chrono::milliseconds interval);

private:
    std::unique_ptr<SQLite::Database> attributes_db;
    const std::string root_path;

    std::mutex file_mutex;
    std::mutex sqlite_mutex;
};

using sptrDatabase = std::shared_ptr<Database>;
