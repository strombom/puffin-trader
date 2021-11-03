#include "Symbols.h"


constexpr auto error_symbol = Symbol{ -1, "ERROR", 0.0, 0.0, 0.0, 0.0, 0.0, 0 };

const Symbol& find_symbol(std::string name) {
    for (const auto& symbol : symbols) {
        if (symbol.name == name) {
            return symbol;
        }
    }
    return error_symbol;
}
