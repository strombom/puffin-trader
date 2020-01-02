
#pragma warning(push, 0)        
#include <torch/torch.h>
#pragma warning(pop)

#include <iostream>

int main() {

    torch::Tensor tensor = torch::eye(5);
    std::cout << tensor << std::endl;
}
