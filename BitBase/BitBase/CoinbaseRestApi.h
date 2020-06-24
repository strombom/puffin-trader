#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "CoinbaseAuthentication.h"


class CoinbaseRestApi
{
public:
    CoinbaseRestApi(void);

    std::tuple<uptrDatabaseTicks, long long> get_aggregate_trades(const std::string& symbol, long long last_id, time_point_ms start_time);

private:
    CoinbaseAuthentication authenticator;

    json11::Json http_get(const std::string& endpoint, json11::Json parameters);
    json11::Json http_post(const std::string& endpoint, json11::Json parameters);
    json11::Json http_delete(const std::string& endpoint, json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrCoinbaseRestApi = std::shared_ptr<CoinbaseRestApi>;
