#pragma once
#include "pch.h"

#include <openssl/hmac.h>


class BinanceAuthentication
{
public:
    BinanceAuthentication(void);
    ~BinanceAuthentication(void);

    long long generate_expiration(std::chrono::seconds timeout);

    std::string authenticate(const std::string& message);

private:
    HMAC_CTX* hmac_ctx;
};
