#include "pch.h"

#include "BitLib/BitBotConstants.h"
#include "BitLib/Logger.h"
#include "BinanceRestApi.h"

#include <iostream>


BinanceRestApi::BinanceRestApi(void)
{

}

std::tuple<sptrTicks, long long> BinanceRestApi::get_aggregate_trades(const std::string& symbol, long long last_trade_id, time_point_ms start_time)
{
    auto end_time = start_time + 1h - 1ms; // Difference between start time and end time must be less than one hour
    
    auto parameters = json11::Json{};
    if (last_trade_id >= 0) {
        parameters = json11::Json::object{
            { "symbol", symbol },
            { "fromId", std::to_string(last_trade_id + 1) },
            { "limit", BitBase::Binance::Tick::max_rows }
        };
    }
    else {
        parameters = json11::Json::object{
            { "symbol", symbol },
            { "startTime", std::to_string(start_time.time_since_epoch().count()) },
            { "endTime", std::to_string(end_time.time_since_epoch().count()) },
            { "limit", BitBase::Binance::Tick::max_rows }
        };
    }

    auto response = http_get("aggTrades", parameters);

    auto ticks = std::make_unique<Ticks>();
    for (auto tick : response.array_items()) {
        const auto timestamp = time_point_ms{ std::chrono::milliseconds{(long long)tick["T"].number_value()} };
        const auto price = std::stof(tick["p"].string_value());
        const auto volume = std::stof(tick["q"].string_value());
        const auto buy = tick["m"].bool_value();
        last_trade_id = (long long)tick["a"].number_value();
        ticks->rows.push_back(Tick{timestamp, price, volume, buy});
    }

    return std::make_tuple(std::move(ticks), last_trade_id);
}

int BinanceRestApi::get_klines(const std::string& symbol, time_point_ms start_time, sptrBinanceKlines binance_klines)
{
    auto parameters = json11::Json::object{
        { "symbol", symbol },
        { "interval", "1m" },
        { "startTime", (double)start_time.time_since_epoch().count() },
        { "limit", 1000 }
    };

    auto response = http_get("klines", parameters);

    if (!response.is_array()) {
        printf("get_klines: response is not an array: %s\n", response.dump().c_str());
    }

    auto count = 0;
    for (const auto &kline : response.array_items()) {
        const auto kline_items = kline.array_items();
        const auto open_time = time_point_ms{ std::chrono::milliseconds{(long long)kline_items[0].number_value()} };
        const auto open = std::stof(kline_items[1].string_value());
        const auto high = std::stof(kline_items[2].string_value());
        const auto low = std::stof(kline_items[3].string_value());
        const auto volume = std::stof(kline_items[5].string_value());
        binance_klines->rows.push_back(BinanceKline{ open_time, open, high, low, volume });
        count++;
    }

    return count;
}

json11::Json BinanceRestApi::http_get(const std::string& endpoint, json11::Json parameters)
{
    auto query_string = std::string{ "?" };
    for (auto &parameter : parameters.object_items()) {
        const auto key = parameter.first;
        const auto value = parameter.second;
        if (value.is_number()) {
            const auto number = value.number_value();
            if (number == std::floor(number)) {
                query_string += key + "=" + std::to_string((long long)value.number_value()) + "&";
            }
            else {
                query_string += key + "=" + std::to_string(value.number_value()) + "&";
            }
        }
        else if (value.is_string()) {
            query_string += key + "=" + value.string_value() + "&";
        }
    }
    query_string.pop_back(); // Remove last "&" or "?" character

    const auto method = "GET";
    const auto url = std::string{ BitBase::Binance::Tick::rest_api_url } + endpoint + query_string;
    const auto version = 11;

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::get);
    req.version(version);
    req.set(boost::beast::http::field::host, BitBase::Binance::Tick::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::accept, "application/json");
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);
    return response;
}

json11::Json BinanceRestApi::http_post(const std::string& endpoint, json11::Json parameters)
{
    const auto method = "POST";
    const auto url = std::string{ BitBase::Binance::Tick::rest_api_url } + endpoint;
    const auto body = parameters.dump();
    const auto expires = authenticator.generate_expiration(BitBase::Binance::Tick::rest_api_auth_timeout);
    const auto sign_message = std::string{ method } + url + std::to_string(expires) + body;
    const auto signature = authenticator.authenticate(sign_message);
    const auto version = 11;

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::post);
    req.version(version);
    req.set(boost::beast::http::field::host, BitBase::Binance::Tick::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::content_type, "application/json");
    req.set(boost::beast::http::field::accept, "application/json");
    req.set("api-expires", std::to_string(expires));
    req.set("api-key", BitBase::Binance::Tick::api_key);
    req.set("api-signature", signature);
    req.body() = body;
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);
    return response;
}

json11::Json BinanceRestApi::http_delete(const std::string& endpoint, json11::Json parameters)
{
    const auto method = "DELETE";
    const auto url = std::string{ BitBase::Binance::Tick::rest_api_url } + endpoint;
    const auto body = parameters.dump();
    const auto expires = authenticator.generate_expiration(BitBase::Binance::Tick::rest_api_auth_timeout);
    const auto sign_message = std::string{ method } + url + std::to_string(expires) + body;
    const auto signature = authenticator.authenticate(sign_message);
    const auto version = 11;

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::delete_);
    req.version(version);
    req.set(boost::beast::http::field::host, BitBase::Binance::Tick::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::content_type, "application/json");
    req.set(boost::beast::http::field::accept, "application/json");
    req.set("api-expires", std::to_string(expires));
    req.set("api-key", BitBase::Binance::Tick::api_key);
    req.set("api-signature", signature);
    req.body() = body;
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);
    return response;
}

const std::string BinanceRestApi::http_request(const boost::beast::http::request<boost::beast::http::string_body>& request)
{
    //try
    {
        // The io_context is required for all I/O
        boost::asio::io_context ioc;

        // The SSL context is required, and holds certificates
        boost::asio::ssl::context ctx(boost::asio::ssl::context::tlsv12_client);

        // This holds the root certificate used for verification
        //load_root_certificates(ctx);

        // Verify the remote server's certificate
        //ctx.set_verify_mode(boost::asio::ssl::verify_peer);

        // These objects perform our I/O
        boost::asio::ip::tcp::resolver resolver(ioc);
        boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);

        // Set SNI Hostname (many hosts need this to handshake successfully)
        //if (!SSL_set_tlsext_host_name(stream.native_handle(), host))
        //{
        //    beast::error_code ec{ static_cast<int>(::ERR_get_error()), net::error::get_ssl_category() };
        //    throw beast::system_error{ ec };
        //}

        // Look up the domain name
        const auto results = resolver.resolve(BitBase::Binance::Tick::rest_api_host, BitBase::Binance::Tick::rest_api_port);

        // Make the connection on the IP address we get from a lookup
        boost::beast::get_lowest_layer(stream).connect(results);

        // Perform the SSL handshake
        stream.handshake(boost::asio::ssl::stream_base::client);

        // Send the HTTP request to the remote host
        boost::beast::http::write(stream, request);

        // This buffer is used for reading and must be persisted
        boost::beast::flat_buffer buffer;

        // Declare a container to hold the response
        boost::beast::http::response<boost::beast::http::dynamic_body> res;

        // Receive the HTTP response
        try {
            boost::beast::http::read(stream, buffer, res);
        }
        catch (boost::system::system_error e) {
            return "";
        }

        const auto body = boost::beast::buffers_to_string(res.body().data());

        // Gracefully close the stream
        boost::beast::error_code ec;
        stream.shutdown(ec);

        return body;
    }
    //catch (std::exception const& e)
    //{
    //    logger.warn("BinanceRestApi::http_request fail (%s)", e.what());
    //}

    return "";
}
