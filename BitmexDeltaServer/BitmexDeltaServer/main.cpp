#include <cstdio>

#include <cpprest/ws_client.h>

int main()
{
    auto client = web::websockets::client::websocket_client{};

    client.connect(U("wss://www.bitmex.com/realtime"));

    auto response = client.receive();

    auto msg = response.get();

    auto body = msg.extract_string().get();

    std::cout << "web: " << body << std::endl;

    printf("hello from BitmexDeltaServer!\n");
    return 0;
}
