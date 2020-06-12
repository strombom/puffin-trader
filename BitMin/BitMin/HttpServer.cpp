#include "pch.h"

#include "HttpServer.h"
#include "BitMinConstants.h"


HttpServer::HttpServer(void) :
    server_thread_running(true)
{

}

void HttpServer::start(void)
{
    server_thread = std::make_unique<std::thread>(&HttpServer::server_worker, this);
}

void HttpServer::shutdown(void)
{
    server_thread_running = false;

    try {
        server_thread->join();
    }
    catch (...) {}
}

void HttpServer::server_worker(void)
{
    while (server_thread_running) {
        std::this_thread::sleep_for(500ms);
    }
}
