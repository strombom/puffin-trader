#pragma once
#include "pch.h"

#include <random>
#include <string>


class Utils
{
public:

    static void save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename);
    static torch::Tensor load_tensor(const std::string& path, const std::string& filename);

    static float random(float min, float max);
    static int random(int min, int max);

private:

};

class RandomRange
{
    // https://stackoverflow.com/questions/288739/generate-random-numbers-uniformly-over-an-entire-range
public:
    RandomRange(const int range_min, const int range_max) :
        random_generator(std::mt19937{ std::random_device{}() }),
        rand_int(range_min, range_max),
        range_min(range_min), range_max(range_max) {}

    RandomRange(const RandomRange& random_range) :
        random_generator(std::mt19937{ std::random_device{}() }),
        rand_int(random_range.range_min, random_range.range_max),
        range_min(random_range.range_min), range_max(random_range.range_max) {}

    int get(void) {
        return rand_int(random_generator);
    }

private:
    int range_min;
    int range_max;
    std::mt19937 random_generator;
    std::uniform_int_distribution<int> rand_int;
};
