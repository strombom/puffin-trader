#include "pch.h"

#include "Logger.h"
#include "DateTime.h"
#include "BitBotConstants.h"
#include "BitmexRestApi.h"


const std::string BitmexRestApi::limit_order(int contracts)
{
    logger.info("BitmexRestApi::limit_order contracts(%d)", contracts);

    json11::Json parameters = json11::Json::object{
        { "symbol", "XBTUSD" },
        { "side", "Buy" },
        { "orderQty", 2 },
        { "price", "18500" },
        { "ordType", "Limit" },
        { "execInst", "ParticipateDoNotInitiate" }
    };

    auto response = http_post(parameters);

    logger.info("Response: %s", response.dump().c_str());

    if (response["orderID"].is_string()) {
        return response["orderID"].string_value();
    }
    else {
        return "";
    }
}

json11::Json BitmexRestApi::http_post(json11::Json parameters)
{
    const auto method = "POST";
    const auto url = std::string{ BitSim::Trader::Bitmex::rest_api_url } + "order";
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
