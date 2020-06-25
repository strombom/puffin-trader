#include "pch.h"

#include "CoinbaseConstants.h"
#include "CoinbaseAuthentication.h"

#include <cstring>
#include <iomanip>


CoinbaseAuthentication::CoinbaseAuthentication(void)
{
    hmac_ctx = HMAC_CTX_new();
}

CoinbaseAuthentication::~CoinbaseAuthentication(void)
{
    HMAC_CTX_free(hmac_ctx);
}

long long CoinbaseAuthentication::generate_expiration(std::chrono::seconds timeout)
{
    const auto now = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now());
    const auto expiration = (now + timeout).time_since_epoch().count();
    return expiration;
}

std::string CoinbaseAuthentication::authenticate(const std::string& message)
{
    HMAC_Init_ex(
        hmac_ctx,
        &Coinbase::RestApi::api_secret[0],
        (int)std::strlen(Coinbase::RestApi::api_secret),
        EVP_sha256(),
        NULL
    );

    HMAC_Update(
        hmac_ctx,
        (const unsigned char*)message.c_str(),
        message.length()
    );

    unsigned char hash[32];
    unsigned int len = 32;
    HMAC_Final(hmac_ctx, hash, &len);

    std::stringstream hash_hex;
    hash_hex << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < len; i++)
    {
        hash_hex << std::hex << std::setw(2) << (unsigned int)hash[i];
    }

    return (hash_hex.str());
}
