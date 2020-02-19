#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"


class RL_Environment
{
public:
    RL_State reset(void);
    RL_Action random_action(void);
};
