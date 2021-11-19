#pragma once
#include "../precompiled_headers.h"


class Uuid
{
public:
    Uuid(uuids::uuid uuid) : uuid(uuid) {}
    Uuid(std::string str) : uuid(uuids::uuid::from_string(str).value()) {}
    Uuid(std::string_view str) : uuid(uuids::uuid::from_string(str).value()) {}
    Uuid(void) : uuid(uuids::uuid()) {}

    const std::string to_string(void) const noexcept;
    bool is_null(void) const noexcept;

private:
    uuids::uuid uuid;

    friend inline bool operator== (Uuid const& lhs, Uuid const& rhs) noexcept;
    friend inline bool operator< (Uuid const& lhs, Uuid const& rhs) noexcept;
};

inline bool operator== (Uuid const& lhs, Uuid const& rhs) noexcept
{
    return lhs.uuid == rhs.uuid;
}

inline bool operator< (Uuid const& lhs, Uuid const& rhs) noexcept
{
    return lhs.uuid < rhs.uuid;
}

class UuidGenerator
{
public:
    UuidGenerator(void);

    Uuid generate(void);

private:
    uuids::uuid_system_generator gen;
};

extern UuidGenerator uuid_generator;
