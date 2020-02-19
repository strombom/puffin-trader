#pragma once
#include "pch.h"

#include "RL_State.h"
#include "RL_Action.h"


class RL_Actor
{
public:
    RL_Action get_action(RL_State state);

};
