#pragma once
#include "pch.h"

#include "BitmexAuthentication.h"


class BitmexRestApi
{
public:
    const std::string limit_order(int contracts);

private:
    BitmexAuthentication authenticator;

    json11::Json http_post(json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};
