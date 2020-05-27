#include "BitmexWebSocket.h"

void BitmexWebSocket::start(void)
{

}

void subscribe(std::shared_ptr<web::websockets::client::websocket_client> client)
{
    /*
    auto response = client->receive();

    auto msg = response.get();

    auto body = msg.extract_string().get();

    std::cout << "conn: " << body << std::endl;

    web::websockets::client::websocket_outgoing_message msg_out;
    msg_out.set_utf8_message("{\"op\": \"subscribe\", \"args\": [\"trade\"]}");
    client->send(msg_out);

    auto client = std::make_shared<web::websockets::client::websocket_client>();

    client->connect(U("wss://www.bitmex.com/realtime"));

    subscribe(client);

    auto running = true;
    while (running) {
        auto response = client->receive();

        auto msg = response.get();

        auto body = msg.extract_string().get();

        std::cout << "rcv: " << body << std::endl;
    }

    */
}


