#include "BitmexWebSocket.h"


BitmexWebSocket::BitmexWebSocket(void) :
    websocket_thread_running(true)
{
    auto config = web::websockets::client::websocket_client_config{};
    client = std::make_unique<web::websockets::client::websocket_client>(config);
}

void BitmexWebSocket::start(void)
{
    client->connect(U(ws_url));
    auto response = client->receive();
    auto msg = response.get();
    auto body = msg.extract_string().get();
    std::cout << "BitmexWebSocket: Websocket connected: " << body << std::endl;

    web::websockets::client::websocket_outgoing_message msg_out;
    msg_out.set_utf8_message("{\"op\": \"subscribe\", \"args\": [\"trade:XBTUSD\", \"trade:ETHUSD\", \"trade:XRPUSD\"]}");
    client->send(msg_out);

    // Start websocket worker
    websocket_thread = std::make_unique<std::thread>(&BitmexWebSocket::websocket_worker, this);
}

void BitmexWebSocket::shutdown(void)
{
    std::cout << "BitmexWebSocket: Shutting down" << std::endl;
    websocket_thread_running = false;
    
    try {
        websocket_thread->join();
    }
    catch (...) {}
}

void BitmexWebSocket::websocket_worker(void)
{
    while (websocket_thread_running) {
        auto response = client->receive();
        auto msg = response.get();
        auto body = msg.extract_string().get();

        std::cout << "BitmexWebSocket: Rcv: " << body << std::endl;
    }
}

