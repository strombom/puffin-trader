#pragma once

#include <array>
#include <string_view>


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
    Symbol{ 0, "BCHUSDT", 0.05, 0.00075, -0.00025, 0.01, 0.01, 600 },
    Symbol{ 1, "ADAUSDT", 0.0001, 0.00075, -0.00025, 1, 1, 240000 },
};
