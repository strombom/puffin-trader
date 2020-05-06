#include "pch.h"
#include "RL_Action.h"
#include "Utils.h"


sptrRL_Action RL_Action::random(void)
{
    const auto discrete_action = Utils::random(0, 2);
    return std::make_shared<RL_Action>(
        //Utils::random(-2.0, 2.0),
        discrete_action == 0,
        discrete_action == 1,
        discrete_action == 2
        );
}

torch::Tensor RL_Action::to_tensor(void) const
{
    return torch::tensor({ 
        //move_side,
        move_not ? 1.0 : 0.0,
        move_up ? 1.0 : 0.0,
        move_down ? 1.0 : 0.0
    });
}

/*
RL_Action RL_Action::random(void)
{
    // Make actions bottom-heavy - more often close to 0 than 1
    return RL_Action{ std::pow(Utils::random(0.0, 1.0), 1.0),    // buy_action
                      std::pow(Utils::random(0.0, 1.0), 10.0),   // buy_size      
                      std::pow(Utils::random(0.0, 1.0), 1.0),    // sell_action 
                      std::pow(Utils::random(0.0, 1.0), 10.0) }; // sell_size     
}

torch::Tensor RL_Action::to_tensor(void) const
{
    return torch::tensor({ buy_action, buy_size, sell_action, sell_size });
}
*/
