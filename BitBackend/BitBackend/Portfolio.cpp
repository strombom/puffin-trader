
#include "Portfolio.h"


Portfolio::Portfolio(void)
{

}

void Portfolio::update_order(Uuid id, const Symbol& symbol, Portfolio::Side side, double price, double qty, std::string status, time_point_us timestamp)
{
    printf("Update %s\n", id.to_string().c_str());
    if (status == "Filled" || status == "Cancelled" || status == "Rejected") {
        if (orders.contains(id)) {
            orders.erase(id);
        }
    }
    else {
        // Created, New, PartiallyFilled, PendingCancel
        orders.insert_or_assign(id, Order{id, symbol, qty, price, side});
    }

    for (const auto& order : orders) {
        printf("Order: %s %s %f %f\n", order.second.uuid.to_string().c_str(), order.second.symbol.name.data(), order.second.price, order.second.qty);
    }
}
