#pragma once
#include "pch.h"

#include "FE_Model.h"
#include "FE_Observations.h"


class FE_Training
{
public:
    FE_Training(sptrFE_Observations observations) :
        observations(observations) {}

    void train(void);
    void test_learning_rate(void);
    void measure_observations(void);

    void save_weights(const std::string& path, const std::string& filename);

private:
    sptrFE_Observations observations;
    RepresentationLearner model{ nullptr };
};

