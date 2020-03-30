#include "pch.h"
#include "RL_Action.h"
#include "Utils.h"


RL_Action RL_Action::random(void)
{
    // Make actions bottom-heavy - more often close to 0 than 1
    return RL_Action{ std::pow(Utils::random(0.0, 1.0), 1.0),    // buy_position  
                      std::pow(Utils::random(0.0, 1.0), 10.0),   // buy_size      
                      std::pow(Utils::random(0.0, 1.0), 1.0),    // sell_position 
                      std::pow(Utils::random(0.0, 1.0), 10.0) }; // sell_size     
}

torch::Tensor RL_Action::to_tensor(void) const
{
    return torch::tensor({ buy_position, buy_size, sell_position, sell_size });
}
