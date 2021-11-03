
#include "Uuid.h"


UuidGenerator uuid_generator;

UuidGenerator::UuidGenerator(void)
{

}

Uuid UuidGenerator::generate(void)
{
    return Uuid{ gen() };
}

const std::string Uuid::to_string(void) const
{
    return uuids::to_string(uuid);
}
