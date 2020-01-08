
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBaseClient.h"
#include "Logger.h"

#include <iostream>


int main() {
    logger.info("BitSim started");

    auto bitbase_client = BitBaseClient();

    bitbase_client.get_intervals();

    //torch::Tensor tensor = torch::eye(5);
    //std::cout << tensor << std::endl;
}
