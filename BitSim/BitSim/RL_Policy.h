#pragma once
#include "pch.h"

#include "RL_SAC.h"

#include <string>


class RL_Policy
{
public:
    RL_Policy(const std::string& filename);

    std::tuple<bool, double> get_action(torch::Tensor feature, double leverage);

private:
    PolicyNetwork policy;
};

using sptrRL_Policy = std::shared_ptr<RL_Policy>;
