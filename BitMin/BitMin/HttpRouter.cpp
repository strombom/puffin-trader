#include "pch.h"

#include "HttpRouter.h"


HttpRouter::HttpRouter(void)
{

}

void HttpRouter::add_route(HttpRouterMethod method, std::string uri, std::function<json11::Json(json11::Json)> callback, std::string arguments)
{

}
