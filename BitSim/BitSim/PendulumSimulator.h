#pragma once
#include "pch.h"

#include "math.h"
#include "RL_State.h"
#include "RL_Action.h"
#include "BitBotConstants.h"


class PendulumSimulatorLogger
{
public:
    PendulumSimulatorLogger(const std::string& filename, bool enabled);

    void log(double angle, double velocity, double reward);

private:
    std::ofstream file;
    bool enabled;

};

class PendulumSimulator
{
public:
    PendulumSimulator(void);

    sptrRL_State reset(const std::string& log_filename);
    sptrRL_State step(sptrRL_Action action, bool last_step);

private:
    sptrRL_State state;

    const double max_speed = 8.0;
    const double max_torque = 2.0;
    const double delta_time = 0.05;
    const double gravity = 10.0;
    const double mass = 1.0;
    const double length = 1.0;

    std::unique_ptr<PendulumSimulatorLogger> logger;
};

using sptrPendulumSimulator = std::shared_ptr<PendulumSimulator>;
