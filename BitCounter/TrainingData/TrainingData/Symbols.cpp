#include "Symbols.h"


constexpr auto error_symbol = Symbol{ -1, "ERROR", 0.0, 0.0, 0.0, 0.0, 0.0, 0 };

const Symbol& string_to_symbol(const char* name) noexcept {
    for (const auto& symbol : symbols) {
        if (symbol.name[0] == name[0] && symbol.name[1] == name[1] && symbol.name[2] == name[2] && symbol.name[3] == name[3]) {
            return symbol;
        }
    }
    return error_symbol;
}

const Symbol& string_to_symbol(std::string name) noexcept {
    for (const auto& symbol : symbols) {
        if (symbol.name[0] == name[0] && symbol.name[1] == name[1] && symbol.name[2] == name[2] && symbol.name[3] == name[3]) {
            return symbol;
        }
    }
    return error_symbol;
}

const Symbol& string_to_symbol(std::string_view name) noexcept {
    for (const auto& symbol : symbols) {
        if (symbol.name[0] == name[0] && symbol.name[1] == name[1] && symbol.name[2] == name[2] && symbol.name[3] == name[3]) {
            return symbol;
        }
    }
    return error_symbol;
}
