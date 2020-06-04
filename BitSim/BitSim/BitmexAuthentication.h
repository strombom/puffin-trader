#pragma once
#include "pch.h"

#include <openssl/hmac.h>


class BitmexAuthentication
{
public:
    BitmexAuthentication(void);
    ~BitmexAuthentication(void);

    long long generate_expiration(std::chrono::seconds timeout);

    std::string authenticate(const std::string& message);

private:
    HMAC_CTX hmac_ctx;
};
