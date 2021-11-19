
#include "Uuid.h"


UuidGenerator uuid_generator;

UuidGenerator::UuidGenerator(void)
{

}

Uuid UuidGenerator::generate(void)
{
    return Uuid{ gen() };
}

const std::string Uuid::to_string(void) const noexcept
{
    return uuids::to_string(uuid);
}

bool Uuid::is_null(void) const noexcept
{
    return uuid.is_nil();
}
