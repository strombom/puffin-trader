#pragma once
#include "pch.h"

class MT_Policy
{
public:
    MT_Policy(void);

    std::tuple<bool, double, double> get_action(torch::Tensor feature);

private:

};

using sptrMT_Policy = std::shared_ptr<MT_Policy>;
