#include "pch.h"

#include <iostream>

#include "HttpServer.h"


int main()
{
    auto http_server = std::make_shared<HttpServer>();

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
