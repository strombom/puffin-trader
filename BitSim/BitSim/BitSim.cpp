
#pragma warning(push, 0)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBase.h"
#include "Logger.h"

#include <iostream>


int main() {
    logger.info("BitSim started");

    auto bitbase = BitBase();

    bitbase.get_intervals();

    //torch::Tensor tensor = torch::eye(5);
    //std::cout << tensor << std::endl;
}
