#pragma once
#include "pch.h"

#include <string>
#include <mutex>
#include <fstream>


class Logger
{
public:
    Logger(void);

    void info(const char* format, ...);
    void warn(const char* format, ...);
    void error(const char* format, ...);

private:
    std::mutex mutex;
};

extern Logger logger;

class CSVLogger
{
public:
    CSVLogger(std::vector<std::string> col_names, std::string file_path);

    template <int array_length>
    CSVLogger(std::array<const char*, array_length> col_names, std::string file_path)
    {
        CSVLogger(std::vector<std::string>{col_names.begin(), col_names.end()}, file_path);
    }

    template <class value_type>
    void append_row(std::vector<value_type> values)
    {
        bool first_row = true;
        for (auto&& value : values) {
            if (first_row) {
                first_row = false;
            } else {
                file << ",";
            }
            file << value;
        }
        file << std::endl;
        file.flush();
    }

    template <class value_type>
    void append_row(std::initializer_list<value_type> values)
    {
        append_row(std::vector<value_type>{ values });
    }

private:
    std::ofstream file;
};
