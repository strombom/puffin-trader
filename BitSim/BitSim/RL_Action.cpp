#include "pch.h"
#include "RL_Action.h"
#include "Utils.h"
#include "BitBotConstants.h"


sptrRL_Action RL_Action::random(void)
{
    const auto disc_action = Utils::random(0, 1);
    //const auto cont_action = Utils::random(-BitSim::BitMex::max_leverage, BitSim::BitMex::max_leverage);
    return std::make_shared<RL_Action>(
        //cont_action,
        disc_action == 0,
        disc_action == 1,
        disc_action == 2,
        disc_action == 3,
        disc_action == 4,
        disc_action == 5
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
            1 * b9 + 2 * b3 + 3 * s1 + 4 * s4 + 5 * s10
            //1 * limit_order + 2 * market_order
        },
        c10::TensorOptions{}.dtype( c10::ScalarType::Long )
    );
}
