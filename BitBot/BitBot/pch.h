#pragma once

#include "BitLib/date/date.h"
#include "BitLib/json11/json11.hpp"

#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4018)
#include "BitLib/csv/csv.h"
#pragma warning(pop)

#include "BitLib/Logger.h"
#include "BitLib/DateTime.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http/write.hpp>
#include <boost/beast/http/parser.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/strand.hpp>
