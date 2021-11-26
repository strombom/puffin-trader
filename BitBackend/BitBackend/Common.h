#pragma once

#include <string>


enum class Side { buy, sell };

Side string_to_side(const std::string& side_str) noexcept;
Side string_to_side(const std::string_view& side_str) noexcept;
std::string side_to_string(Side side) noexcept;
