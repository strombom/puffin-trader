#pragma once
#include "precompiled_headers.h"


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
