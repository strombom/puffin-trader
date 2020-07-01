#include "pch.h"

#include "HttpRouter.h"


HttpRoute::HttpRoute(std::function<json11::Json(json11::Json)> callback, std::string arguments) : 
    callback(callback), arguments(arguments) 
{

}

HttpRoute& HttpRoute::operator=(const HttpRoute& other) {
    callback = other.callback;
    arguments = other.arguments;
    return *this;
}

HttpRouter::HttpRouter(void)
{

}

void HttpRouter::add_route(HttpRouterMethod method, const std::string& path, std::function<json11::Json(json11::Json)> callback, const std::string& arguments)
{
    if (method == HttpRouterMethod::GET) {
        get_routes.emplace(path, HttpRoute{ callback, arguments });
    }
}

std::tuple<bool, std::string> HttpRouter::get_request(const std::string& target)
{
    auto [path, query] = parse_target(target);

    auto response = json11::Json::object{};

    auto found = false;

    if (path == "/directions") {
        response["names"] = json11::Json::object{
            {"bitmex", "Bitmex"},
            {"binance", "Binance"},
            {"coinbase", "Coinbase"}
        };

        response["directions"] = json11::Json::object{
            {"up", json11::Json::array{
                json11::Json::array{{1, 2.6}},
                json11::Json::array{{2, 3.7}}
            }},
            {"down", json11::Json::array{
                json11::Json::array{{0, 1.5}},
                json11::Json::array{{3, 2.8}}
            }},
            {"unknown", json11::Json::array{
                json11::Json::array{{4, 3.9}},
                json11::Json::array{{5, 2.0}}
            }}
        };

        response["prices"] = json11::Json::object{ 
            {"bitmex", json11::Json::array{ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} }},
            {"binance", json11::Json::array{ {1.0, 0.5, 1.0, 1.5, 2.0, 3.0} }},
            {"coinbase", json11::Json::array{ {2.0, 2.5, 2.0, 2.5, 3.0, 2.5} }}
        };
        found = true;
    }

    return std::make_tuple(found, json11::Json{ response }.dump());
}

std::tuple<const std::string, const json11::Json> HttpRouter::parse_target(const std::string& target)
{
    auto path = std::string{};
    auto query = json11::Json::object{};

    auto query_pos = target.find('?');
    query_pos = query_pos == std::string::npos ? target.size() : query_pos;
    
    path = target.substr(0, query_pos);

    if (!test_path_valid_characters(path)) {
        return std::make_tuple("", query);
    }

    
    return std::make_tuple(path, query);
}

bool HttpRouter::test_path_valid_characters(const std::string& path)
{
    return path[std::strspn(path.c_str(), "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-_~!$&'()*+,;=:@")] != 0;
}
