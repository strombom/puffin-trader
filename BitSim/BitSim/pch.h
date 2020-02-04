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

#include "date/date.h"
#include "json11/json11.hpp"
