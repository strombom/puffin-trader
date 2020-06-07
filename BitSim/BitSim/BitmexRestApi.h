#pragma once
#include "pch.h"

#include "BitmexAccount.h"
#include "BitmexAuthentication.h"


class BitmexRestApi
{
public:
    BitmexRestApi(sptrBitmexAccount bitmex_account);

    bool limit_order(int contracts, double price);
    bool delete_all(void);

private:
    sptrBitmexAccount bitmex_account;
    BitmexAuthentication authenticator;

    json11::Json http_post(const std::string& endpoint, json11::Json parameters);
    json11::Json http_delete(const std::string& endpoint, json11::Json parameters);

    const std::string http_request(const boost::beast::http::request<boost::beast::http::string_body>& request);
};

using sptrBitmexRestApi = std::shared_ptr<BitmexRestApi>;
