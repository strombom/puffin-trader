#pragma once

#include <openssl/hmac.h>
#include <chrono>
#include <string>


class ByBitAuthentication
{
public:
    ByBitAuthentication(void);
    ~ByBitAuthentication(void);

    long long generate_expiration(std::chrono::seconds timeout);

    std::string authenticate(const std::string& message);

private:
    HMAC_CTX* hmac_ctx;
};
