#pragma once

#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#include <torch/script.h>
#pragma warning(pop)

#pragma warning(push)
#pragma warning(disable: 4005)
#include <zmq.hpp>
#pragma warning(pop)

#include "BitLib/date/date.h"
#include "BitLib/json11/json11.hpp"
/*
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
*/
