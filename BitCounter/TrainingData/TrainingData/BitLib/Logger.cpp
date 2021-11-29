
#include "Logger.h"


Logger logger;

Logger::Logger(void)
{

}

//#ifdef DEBUG
void Logger::info(const char* format, ...)
{
    /*
    auto lock = std::scoped_lock{ mutex };
    printf("%s INFO: ", date::format("%F %T", std::chrono::system_clock::now()).c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    */
}
//#endif

void Logger::warn(const char* format, ...)
{
    auto lock = std::scoped_lock{ mutex };
    printf("%s WARN: ", date::format("%F %T", std::chrono::system_clock::now()).c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void Logger::error(const char* format, ...)
{
    auto lock = std::scoped_lock{ mutex };
    printf("%s ERR!: ", date::format("%F %T", std::chrono::system_clock::now()).c_str());
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

CSVLogger::CSVLogger(std::vector<std::string> col_names, std::string file_path)
{
    file.open(file_path);
    append_row(col_names);
}
