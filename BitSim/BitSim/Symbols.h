#pragma once

#include <array>


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

    const int idx;
    const std::string_view name;
    const double tick_size;
    const double taker_fee;
    const double maker_fee;
    const double lot_size; 
    const double min_qty; 
    const double max_qty;
};

constexpr const auto symbols = std::array{
    Symbol{ 0, "BTCUSDT", 0.5, 0.00075, -0.00025, 0.001, 0.001, 100 },
    Symbol{ 1, "ETHUSDT", 0.05, 0.00075, -0.00025, 0.01, 0.01, 1000 },
    Symbol{ 2, "EOSUSDT", 0.001, 0.00075, -0.00025, 0.1, 0.1, 50000 },
    Symbol{ 3, "XRPUSDT", 0.0001, 0.00075, -0.00025, 1, 1, 1000000 },
    Symbol{ 4, "BCHUSDT", 0.05, 0.00075, -0.00025, 0.01, 0.01, 600 },
    Symbol{ 5, "LTCUSDT", 0.01, 0.00075, -0.00025, 0.1, 0.1, 2000 },
    Symbol{ 6, "LINKUSDT", 0.001, 0.00075, -0.00025, 0.1, 0.1, 10000 },
    Symbol{ 7, "ADAUSDT", 0.0001, 0.00075, -0.00025, 1, 1, 240000 },
    Symbol{ 8, "DOGEUSDT", 0.0001, 0.00075, -0.00025, 1, 1, 200000 },
    Symbol{ 9, "MATICUSDT", 0.0001, 0.00075, -0.00025, 1, 1, 70000 },
    Symbol{ 10, "ETCUSDT", 0.005, 0.00075, -0.00025, 0.1, 0.1, 2000 },
    Symbol{ 11, "BNBUSDT", 0.05, 0.00075, -0.00025, 0.01, 0.01, 1500 },
    Symbol{ 12, "XLMUSDT", 0.00005, 0.00075, -0.00025, 1, 1, 350000 },
    Symbol{ 13, "THETAUSDT", 0.001, 0.00075, -0.00025, 0.1, 0.1, 15000 },
    Symbol{ 14, "CHZUSDT", 0.00005, 0.00075, -0.00025, 1, 1, 300000 },
};
