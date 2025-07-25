#include "pch.h"

#include "BitLib/Utils.h"

//#include "torch/serialize.h"
#include <iterator>
#include <array>

/*
void Utils::save_tensor(const torch::Tensor& tensor, const std::string& path, const std::string& filename)
{
    torch::save(tensor.cpu(), path + "\\" + filename);
}

torch::Tensor Utils::load_tensor(const std::string &path, const std::string &filename)
{
    auto tensor = std::vector<torch::Tensor>{};
    torch::load(tensor, path + "\\" + filename);
    return tensor[0];
}
*/

UuidGenerator uuid_generator;

UuidGenerator::UuidGenerator(void)
{
    uuid = 0;
}

Uuid UuidGenerator::generate(void)
{
    return uuid++;
    //return Uuid{ gen() };
}

const std::string Uuid::to_string(void) const
{
    return std::to_string(uuid);
    //return uuids::to_string(uuid);
}

double Utils::random(double min, double max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    return std::uniform_real_distribution<double>{min, max}(random_generator);
}

int Utils::random(int min, int max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    return std::uniform_int_distribution<int>{min, max}(random_generator);
}

size_t Utils::random(size_t min, size_t max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    return std::uniform_int_distribution<size_t>{min, max}(random_generator);
}

time_point_ms Utils::random(time_point_ms min, time_point_ms max)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    const auto milliseconds_since_epoch = std::uniform_int_distribution<long long>{ min.time_since_epoch().count(), max.time_since_epoch().count() }(random_generator);
    return time_point_ms{ std::chrono::milliseconds{ milliseconds_since_epoch } };
}

double Utils::random_choice(std::vector<double> choices)
{
    static auto random_generator = std::mt19937{ std::random_device{}() };
    const auto idx = std::uniform_int_distribution<int>{ 0, (int)choices.size() - 1 }(random_generator);
    return choices[idx];
}
