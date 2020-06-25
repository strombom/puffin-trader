#pragma once
#include "pch.h"

#include "BitLib/Ticks.h"
#include "CoinbaseProAuthentication.h"


class CoinbaseProRestApi
{
public:
    CoinbaseProRestApi(void);

    std::tuple<uptrDatabaseTicks, long long> get_aggregate_trades(const std::string& symbol, long long last_id);

private:
    CoinbaseProAuthentication authenticator;

    json11::Json http_get(const std::string& endpoint, json11::Json parameters);

    void load_root_certificates(boost::asio::ssl::context& ctx, boost::system::error_code& ec);
    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrCoinbaseProRestApi = std::shared_ptr<CoinbaseProRestApi>;
