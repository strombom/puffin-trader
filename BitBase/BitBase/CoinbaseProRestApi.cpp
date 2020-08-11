#include "pch.h"

#include "BitLib/Logger.h"
#include "BitLib/DateTime.h"
#include "BitLib/BitBotConstants.h"
#include "CoinbaseProRestApi.h"

#include <iostream>
#include <openssl/ssl.h>


CoinbaseProRestApi::CoinbaseProRestApi(void)
{

}

std::tuple<sptrTicks, long long> CoinbaseProRestApi::get_aggregate_trades(const std::string& symbol, long long last_trade_id)
{
    const auto url = std::string{ "products/" } + symbol + "/trades";
    auto parameters = json11::Json::object{
        { "after", std::to_string(last_trade_id + BitBase::CoinbasePro::Tick::max_rows + 1) }
    };

    auto response = http_get(url, parameters).array_items();
    std::reverse(std::begin(response), std::end(response));

    auto ticks = std::make_unique<Ticks>();
    for (auto tick : response) {
        const auto timestamp = DateTime::to_time_point_ms(tick["time"].string_value(), "%FT%TZ");
        const auto price = std::stof(tick["price"].string_value());
        const auto volume = std::stof(tick["size"].string_value());
        const auto buy = tick["side"].string_value().compare("buy") == 0;
        last_trade_id = (long long)tick["trade_id"].number_value();
        ticks->rows.push_back(Tick{ timestamp, price, volume, buy });
    }

    return std::make_tuple(std::move(ticks), last_trade_id);
}

json11::Json CoinbaseProRestApi::http_get(const std::string& endpoint, json11::Json parameters)
{
    auto query_string = std::string{ "?" };
    for (auto& parameter : parameters.object_items()) {
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
    const auto url = std::string{ BitBase::CoinbasePro::Tick::rest_api_url } + endpoint + query_string;
    const auto version = 11;
    const auto access_timestamp = authenticator.generate_expiration(0s);
    const auto sign_message = std::to_string(access_timestamp) + method + url;
    const auto access_signature = authenticator.authenticate(sign_message);

    boost::beast::http::request<boost::beast::http::string_body> req;
    req.target(url);
    req.method(boost::beast::http::verb::get);
    req.version(version);
    req.set(boost::beast::http::field::host, BitBase::CoinbasePro::Tick::rest_api_host);
    req.set(boost::beast::http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(boost::beast::http::field::accept, "application/json");
    req.set("CB-ACCESS-KEY", BitBase::CoinbasePro::Tick::api_key);
    req.set("CB-ACCESS-SIGN", access_signature);
    req.set("CB-ACCESS-TIMESTAMP", access_timestamp);
    req.set("CB-ACCESS-PASSPHRASE ", BitBase::CoinbasePro::Tick::api_passphrase);
    req.prepare_payload();

    const auto http_response = http_request(req);
    auto error_message = std::string{ "{\"command\":\"error\"}" };
    const auto response = json11::Json::parse(http_response.c_str(), error_message);

    return response;
}

const std::string CoinbaseProRestApi::http_request(const boost::beast::http::request<boost::beast::http::string_body>& request)
{
    try
    {
        boost::beast::error_code ec;

        // The io_context is required for all I/O
        boost::asio::io_context ioc;

        // The SSL context is required, and holds certificates
        boost::asio::ssl::context ctx(boost::asio::ssl::context::tlsv12_client);

        // This holds the root certificate used for verification
        load_root_certificates(ctx, ec);

        // Verify the remote server's certificate
        ctx.set_verify_mode(boost::asio::ssl::verify_peer);

        // These objects perform our I/O
        boost::asio::ip::tcp::resolver resolver(ioc);
        boost::beast::ssl_stream<boost::beast::tcp_stream> stream(ioc, ctx);
        
        // Set SNI Hostname (many hosts need this to handshake successfully)
        if (!SSL_set_tlsext_host_name(stream.native_handle(), BitBase::CoinbasePro::Tick::rest_api_host))
        {
            boost::system::error_code ec{ static_cast<int>(::ERR_get_error()), boost::asio::error::get_ssl_category() };
            throw boost::system::system_error{ ec };
        }

        // Look up the domain name
        const auto results = resolver.resolve(BitBase::CoinbasePro::Tick::rest_api_host, BitBase::CoinbasePro::Tick::rest_api_port);

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
        stream.shutdown(ec);

        return body;
    }
    catch (std::exception const& e)
    {
        logger.warn("CoinbaseProRestApi::http_request fail (%s)", e.what());
    }

    return "";
}

void CoinbaseProRestApi::load_root_certificates(boost::asio::ssl::context& ctx, boost::system::error_code& ec)
{
    // coinbase-com-chain.pem
    const auto cert = std::string{
        "-----BEGIN CERTIFICATE-----\n"
        "MIIDdzCCAl+gAwIBAgIEAgAAuTANBgkqhkiG9w0BAQUFADBaMQswCQYDVQQGEwJJ\n"
        "RTESMBAGA1UEChMJQmFsdGltb3JlMRMwEQYDVQQLEwpDeWJlclRydXN0MSIwIAYD\n"
        "VQQDExlCYWx0aW1vcmUgQ3liZXJUcnVzdCBSb290MB4XDTAwMDUxMjE4NDYwMFoX\n"
        "DTI1MDUxMjIzNTkwMFowWjELMAkGA1UEBhMCSUUxEjAQBgNVBAoTCUJhbHRpbW9y\n"
        "ZTETMBEGA1UECxMKQ3liZXJUcnVzdDEiMCAGA1UEAxMZQmFsdGltb3JlIEN5YmVy\n"
        "VHJ1c3QgUm9vdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMEuyKr\n"
        "mD1X6CZymrV51Cni4eiVgLGw41uOKymaZN+hXe2wCQVt2yguzmKiYv60iNoS6zjr\n"
        "IZ3AQSsBUnuId9Mcj8e6uYi1agnnc+gRQKfRzMpijS3ljwumUNKoUMMo6vWrJYeK\n"
        "mpYcqWe4PwzV9/lSEy/CG9VwcPCPwBLKBsua4dnKM3p31vjsufFoREJIE9LAwqSu\n"
        "XmD+tqYF/LTdB1kC1FkYmGP1pWPgkAx9XbIGevOF6uvUA65ehD5f/xXtabz5OTZy\n"
        "dc93Uk3zyZAsuT3lySNTPx8kmCFcB5kpvcY67Oduhjprl3RjM71oGDHweI12v/ye\n"
        "jl0qhqdNkNwnGjkCAwEAAaNFMEMwHQYDVR0OBBYEFOWdWTCCR1jMrPoIVDaGezq1\n"
        "BE3wMBIGA1UdEwEB/wQIMAYBAf8CAQMwDgYDVR0PAQH/BAQDAgEGMA0GCSqGSIb3\n"
        "DQEBBQUAA4IBAQCFDF2O5G9RaEIFoN27TyclhAO992T9Ldcw46QQF+vaKSm2eT92\n"
        "9hkTI7gQCvlYpNRhcL0EYWoSihfVCr3FvDB81ukMJY2GQE/szKN+OMY3EU/t3Wgx\n"
        "jkzSswF07r51XgdIGn9w/xZchMB5hbgF/X++ZRGjD8ACtPhSNzkE1akxehi/oCr0\n"
        "Epn3o0WC4zxe9Z2etciefC7IpJ5OCBRLbf1wbWsaY71k5h+3zvDyny67G7fyUIhz\n"
        "ksLi4xaNmjICq44Y3ekQEe5+NauQrz4wlHrQMz2nZQ/1/I6eYs9HRCwBXbsdtTLS\n"
        "R9I4LtD+gdwyah617jzV/OeBHRnDJELqYzmp\n"
        "-----END CERTIFICATE-----\n"
    };

    ctx.add_certificate_authority(boost::asio::buffer(cert.data(), cert.size()), ec);
}
