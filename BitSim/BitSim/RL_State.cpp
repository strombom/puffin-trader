#include "pch.h"
#include "RL_State.h"


void RL_State::set_done(void)
{
    done = true;
}

bool RL_State::is_done(void)
{
    return done;
}
