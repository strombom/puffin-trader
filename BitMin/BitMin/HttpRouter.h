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

class HttpRoute
{
public:
    HttpRoute(std::function<json11::Json(json11::Json)> callback, std::string arguments);
    
    HttpRoute& operator=(const HttpRoute& other);

    std::function<json11::Json(json11::Json)> callback;
    std::string arguments;
};

class HttpRouter
{
public:
    HttpRouter(void);

    void add_route(HttpRouterMethod method, const std::string& path, std::function<json11::Json(json11::Json)> callback, const std::string& arguments);
    
    std::tuple<bool, std::string> get_request(const std::string &target);

private:
    std::map<std::string, HttpRoute> get_routes;

    bool test_path_valid_characters(const std::string& path);
    std::tuple<const std::string, const json11::Json> parse_target(const std::string& target);
};
