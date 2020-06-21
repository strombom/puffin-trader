#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "BinanceAuthentication.h"


class BinanceRestApi
{
public:
    BinanceRestApi(void);

    uptrDatabaseTicks get_aggregate_trades(const std::string& symbol, time_point_ms start_time);

private:
    BinanceAuthentication authenticator;

    json11::Json http_get(const std::string& endpoint, json11::Json parameters);
    json11::Json http_post(const std::string& endpoint, json11::Json parameters);
    json11::Json http_delete(const std::string& endpoint, json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrBinanceRestApi = std::shared_ptr<BinanceRestApi>;
