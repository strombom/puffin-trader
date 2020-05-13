#include "pch.h"
#include "RL_Action.h"
#include "Utils.h"


sptrRL_Action RL_Action::random(void)
{
    const auto disc_action_x = Utils::random(0, 2);
    const auto cont_action_y = Utils::random(-1.0, 1.0);
    return std::make_shared<RL_Action>(
        disc_action_x == 0,
        disc_action_x == 1,
        disc_action_x == 2,
        cont_action_y
        );
}

torch::Tensor RL_Action::to_tensor_cont(void) const
{
    return torch::tensor({
        y_move
        });
}

torch::Tensor RL_Action::to_tensor_disc(void) const
{
    return torch::tensor({ 
        1 * x_move_left + 2 * x_move_right
        },
        c10::TensorOptions{}.dtype( c10::ScalarType::Long )
    );
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
