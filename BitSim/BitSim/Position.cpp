#include "pch.h"
#include "Position.h"


Position::Position(time_point_ms created, int delta_idx, sptrOrder order) :
    created(created), uuid(uuid_generator.generate()), state(State::Opening), symbol(order->symbol), delta_idx(delta_idx), 
    created_price(order->price), created_amount(order->quantity), filled_price(0), filled_amount(0), take_profit(0), stop_loss(0), order(order)
{

}
