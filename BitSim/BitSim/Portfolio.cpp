#include "pch.h"
#include "Portfolio.h"
#include "BitLib/Logger.h"


void Portfolio::print_portfolio(time_point_ms timestamp)
{
    auto str = DateTime::to_string(timestamp);
    str += " Hodlings " + std::to_string(simulator.get_total_equity()) + "USDT";
    for (const auto& symbol : symbols) {
        const auto quantity = simulator.get_wallet(symbol);
        if (quantity > 0) {
            str += " " + std::to_string(quantity * simulator.get_mark_price(symbol)) + " " + std::string{ symbol.name };
        }
    }
    logger.info("%s", str.c_str());

    //logger.info("%s Position(%s) Close %s @%.5f", DateTime::to_string(timestamp).c_str(), position.uuid.to_string().c_str(), position.symbol.name.data(), price);

}

void Portfolio::set_mark_prices(const Klines& klines)
{
    simulator.set_mark_prices(klines);
}

void Portfolio::evaluate_positions(time_point_ms timestamp)
{

}

bool Portfolio::has_available_position(const Symbol& symbol)
{
    return get_position_count() < BitSim::Portfolio::total_capacity && get_position_count(symbol) < BitSim::Portfolio::symbol_capacity;
}

bool Portfolio::has_available_order(const Symbol& symbol)
{
    return positions.size() < BitSim::Portfolio::total_capacity && get_position_count(symbol) + get_order_count(symbol) < BitSim::Portfolio::symbol_capacity;
}

void Portfolio::cancel_oldest_order(time_point_ms timestamp, const Symbol& symbol)
{
    const auto symbol_position_count = get_position_count(symbol);
    const auto symbol_order_count = get_order_count(symbol);

    auto oldest_position = (Position*)(nullptr);
    for (auto& position : positions) {
        if (position.state == Position::State::Opening && (position.symbol == symbol || symbol_position_count + symbol_order_count < BitSim::Portfolio::symbol_capacity) && (oldest_position == nullptr|| position.created > oldest_position->created)) {
            oldest_position = &position;
        }
    }
    if (oldest_position == nullptr) {
        return;
    }
    oldest_position->order->cancel = true;
    //logger.info("%s Position(%s) Cancel %s", DateTime::to_string(timestamp).c_str(), oldest_position->uuid.to_string().c_str(), oldest_position->symbol.name.data());

    simulator.cancel_orders();

    positions.erase(
        std::remove_if(
            positions.begin(),
            positions.end(),
            [](const Position& position) {
                return position.state == Position::State::Opening && position.order->state == Order::State::Canceled;
            }
        ),
        positions.end()
    );
}

void Portfolio::place_limit_order(time_point_ms timestamp, const Symbol& symbol, int delta_idx, double position_size)
{
    // TODO: set significant digits for price
    const auto price = simulator.get_mark_price(symbol);
    const auto quantity = int(position_size / symbol.min_qty) * symbol.min_qty;

    if (quantity < 0) {
        auto a = 0;
    }

    auto order = simulator.limit_order(timestamp, symbol, price, quantity);

    if (order == nullptr) {
        //logger.info("%s Position(%s) Failed to place limit order %.5f %.5f", DateTime::to_string(timestamp).c_str(), symbol.name.data(), price, quantity);
        return;
    }

    if (order->amount == 0) {
        auto a = 0;
    }

    positions.push_back({ timestamp, delta_idx, order });
    const auto& position = positions.back();

     logger.info("%s Position(%s) Limit order %s %.5f %.5f", DateTime::to_string(timestamp).c_str(), position.uuid.to_string().c_str(), symbol.name.data(), price, quantity);
}

void Portfolio::evaluate_orders(time_point_ms timestamp, const Klines& klines)
{
    // Update limit orders
    for (auto& position : positions) {
        if (position.state == Position::State::Active) {
            const auto mark_price = simulator.get_mark_price(position.symbol);
            if (mark_price < position.stop_loss || mark_price > position.take_profit) {
                // Close position by placing a sell order
                const auto price = simulator.get_mark_price(position.symbol);
                position.state = Position::State::Closing;
                position.order = simulator.limit_order(timestamp, position.symbol, price, -position.amount);
                //logger.info("%s Position(%s) Close %s @%.5f", DateTime::to_string(timestamp).c_str(), position.uuid.to_string().c_str(), position.symbol.name.data(), price);

            }
        }

        if (position.state == Position::State::Opening || position.state == Position::State::Closing) {
            // Adjust limit order price
            // TODO: Cancel if price/time is too far off
            if (position.order->side == Order::Side::Buy) {
                position.order->price = simulator.get_mark_price(position.symbol) - position.symbol.tick_size;
            }
            else {
                position.order->price = simulator.get_mark_price(position.symbol) + position.symbol.tick_size;
            }
            simulator.adjust_order_volumes();
        }
    }

    simulator.evaluate_orders(timestamp, klines);

    for (auto& position : positions) {
        if (position.state == Position::State::Opening && position.order->state == Order::State::Filled) {
            position.state = Position::State::Active;
            //position.amount = position.order->amount;
            position.filled_price = position.order->price;
            position.take_profit = position.filled_price * BitBot::Trading::take_profit[position.delta_idx];
            position.stop_loss = position.filled_price * BitBot::Trading::stop_loss[position.delta_idx];
            
            //logger.info("%s Position(%s) Filled %s @%.5f", DateTime::to_string(timestamp).c_str(), position.uuid.to_string().c_str(), position.symbol.name.data(), position.order->price);

            position.order = nullptr;
        }
        else if (position.state == Position::State::Closing && position.order->state == Order::State::Filled) {

            //logger.info("%s Position(%s) Closed %s @%.5f", DateTime::to_string(timestamp).c_str(), position.uuid.to_string().c_str(), position.symbol.name.data(), position.order->price);

            if (position.amount != -position.order->amount) {
                auto a = 0;
            }

            print_portfolio(timestamp);

            position.state = Position::State::Closed;
        }
    }

    positions.erase(
        std::remove_if(
            positions.begin(),
            positions.end(),
            [](const Position& position) {
                return position.state == Position::State::Closed;
            }
        ),
        positions.end()
    );
}

double Portfolio::get_equity(void) const
{
    double equity = simulator.get_wallet_usdt();
    for (const auto& symbol : symbols) {
        equity += simulator.get_wallet(symbol) * simulator.get_mark_price(symbol);
    }
    return equity;
}

double Portfolio::get_cash(void) const
{
    return simulator.get_wallet_usdt();
}

inline int Portfolio::get_position_count(void)
{
    return (int)std::count_if(positions.begin(), positions.end(), [](const auto& position) {return position.state != Position::State::Opening; });
}

inline int Portfolio::get_position_count(const Symbol& symbol)
{
    return (int)std::count_if(positions.begin(), positions.end(), [symbol](const auto& position) {return position.symbol == symbol && position.state != Position::State::Opening; });
}

inline int Portfolio::get_order_count(const Symbol& symbol)
{
    return (int)std::count_if(positions.begin(), positions.end(), [symbol](const auto& position) {return position.symbol == symbol && position.state == Position::State::Opening; });
}
