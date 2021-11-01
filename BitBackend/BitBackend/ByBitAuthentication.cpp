
#include "ByBitAuthentication.h"

ByBitAuthentication::ByBitAuthentication(void)
{

}

ByBitAuthentication::~ByBitAuthentication(void)
{

}

long long ByBitAuthentication::generate_expiration(std::chrono::seconds timeout)
{
    return 0;
}

std::string ByBitAuthentication::authenticate(const std::string& message)
{
    return std::string{ "" };
}
