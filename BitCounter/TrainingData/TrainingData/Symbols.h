#pragma once

#include <array>
#include <string>


struct Symbol {
    constexpr Symbol(
        int idx, 
        const std::string_view name, 
        double tick_size, 
        double taker_fee, 
        double maker_fee, 
        double lot_size, 
        double min_qty, 
        double max_qty) 
    : 
        idx(idx), 
        name(name), 
        tick_size(tick_size), 
        taker_fee(taker_fee), 
        maker_fee(maker_fee), 
        lot_size(lot_size), 
        min_qty(min_qty), 
        max_qty(max_qty) {}

    int idx;
    std::string_view name;
    double tick_size;
    double taker_fee;
    double maker_fee;
    double lot_size; 
    double min_qty; 
    double max_qty;

    bool operator==(const Symbol& other) const {
        return idx == other.idx;
    }
};

constexpr const auto symbols = std::array{
    Symbol{ 0, "BTCUSDT", 0.5, 0.00075, -0.00025, 0.001, 0.001, 100 },
    //Symbol{ 1, "THETAUSDT", 0.001, 0.00075, -0.00025, 0.1, 0.1, 15000 },
};

const Symbol& string_to_symbol(const char* name) noexcept;
const Symbol& string_to_symbol(std::string name) noexcept;
const Symbol& string_to_symbol(std::string_view name) noexcept;
