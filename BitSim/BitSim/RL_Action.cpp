#include "pch.h"
#include "RL_Action.h"
#include "BitLib/Utils.h"
#include "BitLib/BitBotConstants.h"


sptrRL_Action RL_Action::random(void)
{
    const auto disc_action = Utils::random(0, 1);
    const auto cont_action_0 = Utils::random(-BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    const auto cont_action_1 = Utils::random(-BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    return std::make_shared<RL_Action>(
        //cont_action,
        disc_action == 1,
        cont_action_0,
        cont_action_1
    );
}

/*
torch::Tensor RL_Action::to_tensor_cont(void) const
{
    return torch::tensor({
        leverage
        });
}
*/

torch::Tensor RL_Action::to_tensor_disc(void) const
{
    return torch::tensor({
            1 * buy
            //1 * limit_order + 2 * market_order
        },
        c10::TensorOptions{}.dtype( c10::ScalarType::Long )
    );
}
