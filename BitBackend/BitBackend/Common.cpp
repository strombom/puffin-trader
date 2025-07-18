#include "Common.h"


Side string_to_side(const std::string& side_str) noexcept
{
    if (side_str[0] == 'B') {
        return Side::buy;
    }
    else {
        return Side::sell;
    }
}

Side string_to_side(const std::string_view& side_str) noexcept
{
    if (side_str[0] == 'B') {
        return Side::buy;
    }
    else {
        return Side::sell;
    }
}

std::string side_to_string(Side side) noexcept
{
    if (side == Side::buy) {
        return "Buy";
    }
    else {
        return "Sell";
    }
}
