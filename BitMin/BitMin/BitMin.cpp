#include "pch.h"

#include <iostream>

#include "HttpServer.h"
#include "HttpRouter.h"
#include "DirectionEstimation.h"


int main()
{
    auto direction_estimation = DirectionEstimation{};
    auto http_router = std::make_shared<HttpRouter>();

    const auto get_directions = std::bind(&DirectionEstimation::get_directions, &direction_estimation, std::placeholders::_1);
    http_router->add_route(HttpRouterMethod::GET, "/directions", get_directions, "BITMEX_XBTUSD");

    auto http_server = std::make_shared<HttpServer>(http_router);

    http_server->start();

    while (true) {
        auto command = std::string{};
        std::cin >> command;
        if (command.compare("q") == 0) {
            break;
        }
    }

    http_server->shutdown();

    return 0;
}
