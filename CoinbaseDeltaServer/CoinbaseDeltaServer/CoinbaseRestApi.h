#pragma once
#include "pch.h"

#include "TickData.h"
#include "CoinbaseAuthentication.h"

#include "BitLib/json11/json11.hpp"

#include <boost/beast/core.hpp>
#include <boost/beast/http/write.hpp>
#include <boost/beast/http/parser.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/strand.hpp>


class CoinbaseRestApi
{
public:
    CoinbaseRestApi(void);

    std::tuple<sptrTickData, long long> get_aggregate_trades(const std::string& symbol, long long last_id, time_point_ms start_time);

private:
    CoinbaseAuthentication authenticator;

    json11::Json http_get(const std::string& endpoint, json11::Json parameters);

    void load_root_certificates(boost::asio::ssl::context& ctx, boost::system::error_code& ec);
    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrCoinbaseRestApi = std::shared_ptr<CoinbaseRestApi>;
