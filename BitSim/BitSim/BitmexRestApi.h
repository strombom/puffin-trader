#pragma once
#include "pch.h"

#include "BitmexAccount.h"
#include "BitmexAuthentication.h"


class BitmexRestApi
{
public:
    BitmexRestApi(sptrBitmexAccount bitmex_account);

    const std::string limit_order(int contracts);

private:
    sptrBitmexAccount bitmex_account;
    BitmexAuthentication authenticator;

    json11::Json http_post(json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrBitmexRestApi = std::shared_ptr<BitmexRestApi>;
