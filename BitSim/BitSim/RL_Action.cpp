#include "pch.h"
#include "RL_Action.h"
#include "BitLib/Utils.h"
#include "BitLib/BitBotConstants.h"


sptrRL_Action RL_Action::random(void)
{
    const auto disc_action = Utils::random(0, 1);
    const auto cont_action_0 = Utils::random(0.0, 1.0);
    return std::make_shared<RL_Action>(
        disc_action == 0 ? RL_Action_Direction::dir_long : RL_Action_Direction::dir_short,
        cont_action_0
    );
}

torch::Tensor RL_Action::to_tensor_cont(void) const
{
    return torch::tensor({
        stop_loss
        });
}

torch::Tensor RL_Action::to_tensor_disc(void) const
{
    return torch::tensor({
            1 * direction
            //1 * limit_order + 2 * market_order
        },
        c10::TensorOptions{}.dtype( c10::ScalarType::Long )
    );
}
