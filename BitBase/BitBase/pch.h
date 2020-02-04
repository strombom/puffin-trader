#pragma once

#include "date/date.h"
#include "curl/curl.h"
#include "json11/json11.hpp"
#include "SQLiteCpp/SQLiteCpp.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/copy.hpp>

#pragma warning(push)
#pragma warning(disable: 4005)
#include <zmq.hpp>
#pragma warning(pop)
