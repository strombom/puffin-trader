#pragma once
#include "pch.h"

#include "BitLib/json11/json11.hpp"

#include <functional>


enum HttpRouterMethod
{
    GET,
    PUT,
    POST,
    DELETE
};

class HttpRouter
{
public:
    HttpRouter(void);

    void add_route(HttpRouterMethod method, std::string uri, std::function<json11::Json(json11::Json)> callback, std::string arguments);
private:

};
