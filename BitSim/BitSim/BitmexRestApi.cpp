#include "pch.h"

#include "Logger.h"
#include "DateTime.h"
#include "BitmexRestApi.h"
#include "BitBotConstants.h"


BitmexRestApi::BitmexRestApi(sptrBitmexAccount bitmex_account) :
    bitmex_account(bitmex_account)
{

}

bool BitmexRestApi::limit_order(int contracts, double price)
{
    logger.info("BitmexRestApi::limit_order contracts(%d)", contracts);

    const auto side = contracts > 0 ? "Buy" : "Sell";

    json11::Json parameters = json11::Json::object{
        { "symbol", "XBTUSD" },
        { "side", side },
        { "orderQty", std::abs(contracts) },
        { "price", price },
        { "ordType", "Limit" },
        { "execInst", "ParticipateDoNotInitiate" }
    };

    auto response = http_post("order", parameters);

    //logger.info("Response: %s", response.dump().c_str());

    if (response["ordStatus"].string_value() == "New" ||
        response["ordStatus"].string_value() == "Partially filled") {
        
        const auto order_id = response["orderID"].string_value();
        const auto symbol = response["symbol"].string_value();
        const auto timestamp = DateTime::to_time_point_ms(response["timestamp"].string_value(), "%FT%TZ");
        const auto buy = (response["side"].string_value() == "Buy");
        const auto order_size = response["orderQty"].int_value();
        const auto price = response["price"].number_value();

        bitmex_account->insert_order(symbol, order_id, timestamp, buy, order_size, price);
        

        return true;
    }
    else if (response["ordStatus"].string_value() == "Filled") {
        return true;
    }
    else {
        return false;
    }
}

bool BitmexRestApi::delete_all(void)
{
    logger.info("BitmexRestApi::delete_all");

    json11::Json parameters = json11::Json::object{
        { "symbol", "XBTUSD" }
    };

    auto response = http_delete("order/all", parameters);

    //logger.info("Response: %s", response.dump().c_str());

    auto fail = false;
    for (const auto& data : response.array_items()) {
        if (data["ordStatus"].string_value() == "Canceled") {
            const auto order_id = data["orderID"].string_value();
            bitmex_account->delete_order(order_id);
        }
        else {
            fail = true;
        }
    }

    const auto success = !fail;
    return success;
}

json11::Json BitmexRestApi::http_post(const std::string& endpoint, json11::Json parameters)
{
    const auto method = "POST";
    const auto url = std::string{ BitSim::Trader::Bitmex::rest_api_url } + endpoint;
    const auto body = parameters.dump();
    const auto expires = authenticator.generate_expiration(BitSim::Trader::Bitmex::rest_api_auth_timeout);
    const auto sign_message = std::string{ method } + url + std::to_string(expires) + body;
    const auto signature = authenticator.authenticate(sign_message);
    const auto version = 11;

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::post);
    req.version(version);
    req.set(boost::beast::http::field::host, BitSim::Trader::Bitmex::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::content_type, "application/json");
    req.set(boost::beast::http::field::accept, "application/json");
    req.set("api-expires", expires);
    req.set("api-key", BitSim::Trader::Bitmex::api_key);
    req.set("api-signature", signature);
    req.body() = body;
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);
    return response;
}

json11::Json BitmexRestApi::http_delete(const std::string& endpoint, json11::Json parameters)
{
    const auto method = "DELETE";
    const auto url = std::string{ BitSim::Trader::Bitmex::rest_api_url } + endpoint;
    const auto body = parameters.dump();
    const auto expires = authenticator.generate_expiration(BitSim::Trader::Bitmex::rest_api_auth_timeout);
    const auto sign_message = std::string{ method } + url + std::to_string(expires) + body;
    const auto signature = authenticator.authenticate(sign_message);
    const auto version = 11;

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::delete_);
    req.version(version);
    req.set(boost::beast::http::field::host, BitSim::Trader::Bitmex::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::content_type, "application/json");
    req.set(boost::beast::http::field::accept, "application/json");
    req.set("api-expires", expires);
    req.set("api-key", BitSim::Trader::Bitmex::api_key);
    req.set("api-signature", signature);
    req.body() = body;
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);
    return response;
}

const std::string BitmexRestApi::http_request(const boost::beast::http::request<boost::beast::http::string_body>& request)
{

    //try
    //{

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

        // Look up the domain name
        const auto results = resolver.resolve(BitSim::Trader::Bitmex::rest_api_host, BitSim::Trader::Bitmex::rest_api_port);

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
        boost::beast::http::read(stream, buffer, res);

        const auto body = boost::beast::buffers_to_string(res.body().data());

        // Gracefully close the stream
        boost::beast::error_code ec;
        stream.shutdown(ec);

        return body;


        /*
        if (ec == boost::asio::error::eof)
        {
            // Rationale:
            // http://stackoverflow.com/questions/25587403/boost-asio-ssl-async-shutdown-always-finishes-with-an-error
            ec = {};
        }

        std::cout << "Err: " << ec.message().c_str() << std::endl;
        std::cout << "Err: " << ec.value() << std::endl;

        if (ec) {


            return "";
            //throw boost::beast::system_error{ ec };
        }
        */

        //auto parser = boost::beast::http::response_parser<boost::beast::http::string_body>{};
        //parser.put(res, ec);
        //parser.put_eof(ec);
        //auto message = parser.get();
        //auto body = message.body();

        //std::cout << "Body: " << body.c_str() << std::endl;


        // If we get here then the connection is closed gracefully
    //}
    //catch (std::exception const& e)
    //{
    //    logger.warn("BitmexRestApi::http_request fail (%s) (%s)", e.what(), target.c_str());
        //std::cerr << "Error: " << e.what() << std::endl;
    //}

    return "";
}
