
#include "BitmexWebSocket.h"
#include "DateTime.h"


BitmexWebSocket::BitmexWebSocket(sptrTickData tick_data) :
    tick_data(tick_data),
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

bool BitmexWebSocket::json_test_field(const web::json::value& data, const std::string& name, const std::string& value)
{
    return data.has_field(U(name)) && data.at(U(name)).as_string().compare(U(value)) == 0;
}

void BitmexWebSocket::websocket_worker(void)
{
    while (websocket_thread_running) {
        const auto response = client->receive().get();
        const auto body = response.extract_string().get();
        const auto data = web::json::value::parse(U(body));
        
        if (json_test_field(data, "action", "insert") && json_test_field(data, "table", "trade"))
        {
            const auto ticks = data.at(U("data")).as_array();

            for (auto tick : ticks) {
                const auto symbol = tick.at(U("symbol")).as_string();
                const auto price = tick.at(U("price")).as_double();
                const auto volume = tick.at(U("size")).as_double();
                const auto buy = tick.at(U("side")).as_string().compare(U("Buy")) == 0;
                const auto timestamp = DateTime::to_time_point_ms(tick.at(U("timestamp")).as_string(), "%FT%TZ");

                std::wcout << "BitmexWebSocket: Insert table: " <<
                    "timestamp(" << DateTime::to_string_iso_8601(timestamp) << ") " <<
                    "symbol(" << symbol.c_str() << ") " <<
                    "price(" << price << ") " <<
                    "volume(" << volume << ") " <<
                    "buy(" << buy << ") " << std::endl;

                tick_data->append(symbol, timestamp, (float) price, (float) volume, buy);
            }
        }
        else
        {
            std::wcout << "BitmexWebSocket: Rcv: " << data.to_string().c_str() << std::endl;
        }

    }
}
