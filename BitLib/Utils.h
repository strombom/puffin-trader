#pragma once
#include "pch.h"

#include "BitLib/DateTime.h"

#define UUID_SYSTEM_GENERATOR
#include "stduuid/uuid.h"

#include <random>
#include <string>


class Utils
{
public:

    //static void save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename);
    //static torch::Tensor load_tensor(const std::string &path, const std::string &filename);

    static double random(double min, double max);
    static int random(int min, int max);
    static size_t random(size_t min, size_t max);
    static time_point_ms random(time_point_ms min, time_point_ms max);
    static double random_choice(std::vector<double> choices);

private:

};

class Uuid
{
public:
    Uuid(uuids::uuid uuid) : uuid(uuid) {}

    const std::string to_string(void) const;

private:
    uuids::uuid uuid;
};

class UuidGenerator
{
public:
    UuidGenerator(void) {}

    Uuid generate(void);

private:
    uuids::uuid_system_generator gen;
};

extern UuidGenerator uuid_generator;

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
