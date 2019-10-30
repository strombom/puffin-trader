#include "Logger.h"
#include "Database.h"

Database::Database(const std::string &root_path)
{
    logger.info("Hello Database %d", 12);
}
