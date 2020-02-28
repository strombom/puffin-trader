#include "pch.h"
#include "RL_Action.h"
#include "Utils.h"


RL_Action RL_Action::random(void)
{
    return RL_Action{ Utils::random(0.0, 1.0), 
                      Utils::random(0.0, 1.0), 
                      Utils::random(0.0, 1.0),
                      Utils::random(0.0, 1.0) };
}
