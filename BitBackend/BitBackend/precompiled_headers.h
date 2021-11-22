#pragma once

#include "BitLib/date/date.h"
#include "BitLib/stduuid/uuid.h"
#include "BitLib/json11/json11.hpp"

#include <ios>
#include <mutex>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <string>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdarg.h>
#include <iostream>
#include <simdjson.h> // vcpkg install simdjson
#include <semaphore.h>
#include <openssl/ssl.h> // vcpkg install openssl
#include <openssl/hmac.h>
#include <boost/asio/strand.hpp> // vcpkg install boost-asio
#include <boost/beast/websocket.hpp> // vcpkg install boost-beast
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/ssl/ssl_stream.hpp>
#include <boost/beast/core/tcp_stream.hpp>
