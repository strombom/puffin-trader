#pragma once
#include "pch.h"

#include "math.h"
#include "RL_State.h"
#include "RL_Action.h"
#include "BitBotConstants.h"


class CartpoleSimulatorLogger
{
public:
    CartpoleSimulatorLogger(const std::string& filename, bool enabled);

    void log(double cart_position, double cart_velocity, double pole_angle, double pole_velocity);

private:
    std::ofstream file;
    bool enabled;

};

class CartpoleSimulator
{
public:
    CartpoleSimulator(void);

    RL_State reset(const std::string& log_filename);
    RL_State step(const RL_Action& action);

private:
    RL_State state;

    const double gravity = 9.8;
    const double mass_cart = 1.0;
    const double mass_pole = 0.1;
    const double length = 0.5;
    const double force_mag = 10.0;
    const double tau = 0.02;
    const double theta_threshold = 12 * 2 * M_PI / 360;
    const double x_threshold = 2.4;

    const double polemass_length = mass_pole * length;
    const double total_mass = mass_cart + mass_pole;

    std::unique_ptr<CartpoleSimulatorLogger> logger;
};

using sptrCartpoleSimulator = std::shared_ptr<CartpoleSimulator>;
