#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "BinanceAuthentication.h"
#include "BinanceKlines.h"


class BinanceRestApi
{
public:
    BinanceRestApi(void);

    std::tuple<sptrTicks, long long> get_aggregate_trades(const std::string& symbol, long long last_trade_id, time_point_ms start_time);
    int get_klines(const std::string& symbol, time_point_ms start_time, sptrBinanceKlines binance_klines);

private:
    BinanceAuthentication authenticator;

    json11::Json http_get(const std::string& endpoint, json11::Json parameters);
    json11::Json http_post(const std::string& endpoint, json11::Json parameters);
    json11::Json http_delete(const std::string& endpoint, json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrBinanceRestApi = std::shared_ptr<BinanceRestApi>;
