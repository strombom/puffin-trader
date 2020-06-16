#!/usr/bin/env python3

import argparse
from math import ceil, floor


sign = lambda amt: -1 if amt<0 else (1 if amt>0 else 0)


class dobj:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def target_liquidation(qty, price):
    round_up = sign(qty) # make positive for round up
    s = round_up * ceil(round_up * price / qty)
    l = s if s else sign(qty / price)
    return round(-100000000 / l * 2) * 0.5


def tousd(qty, price):
    a = round(-100000000 / price)
    return round(qty * a)


def step_pos(qty, usd, current_qty, cost):
    return round(cost * min(1, -qty / current_qty)) + round(usd * min(1, -current_qty / qty)) if sign(qty) == -sign(current_qty) else 0


def new_pos_data(inst, pos, qty, price):
    total_qty = pos.current_qty + qty
    mark_usd = tousd(total_qty, inst.mark_price)
    #if div: pos.init_margin_req = 1 / div
    order_usd = tousd(qty, price)
    total_usd = pos.current_cost + order_usd
    new_pos_cross = pos.pos_cross - step_pos(qty, 0, pos.current_qty, pos.pos_cross);
    #if div: new_pos_cross += Math.max(0, -pos.unrealised_pnl)
    additional_cost = step_pos(qty, order_usd, pos.current_qty, pos.current_cost - pos.realised_cost)
    net_total_usd = total_usd - (pos.realised_cost + additional_cost)
    init_margin_usd = abs(net_total_usd) * pos.init_margin_req
    additional_commission = round((abs(net_total_usd) + max(0, new_pos_cross + init_margin_usd)) * pos.commission)
    new_unrealized_pnl = mark_usd - net_total_usd
    return dobj(new_current_qty = total_qty,
                new_mark_value = mark_usd,
                new_pos_cross = new_pos_cross,
                new_pos_init = init_margin_usd,
                new_unrealised_pnl = new_unrealized_pnl,
                new_pos_comm = additional_commission,
                new_pos_cost = net_total_usd,
                new_realised_pnl = pos.realised_pnl - additional_cost - round(abs(order_usd) * pos.commission),
                new_maint_margin = max(0, new_pos_cross + init_margin_usd + additional_commission + new_unrealized_pnl))


def liquidation_price(inst, pos, marg, qty, price):
    d = new_pos_data(inst, pos, qty, price)
    new_mark_sign = sign(d.new_mark_value)
    margin_req = pos.maint_margin_req
    if margin_req > inst.maint_margin:
        f = pos.risk_value - abs(pos.mark_value) + abs(d.new_mark_value);
        p = d.new_pos_comm + abs(d.new_pos_cost) * min(pos.init_margin_req, min(margin_req, inst.maint_margin * (1 + ceil((f - inst.risk_limit) / inst.risk_step))) + max(0, inst.funding_rate * sign(d.new_current_qty)))
    else:
        p = d.new_pos_comm + abs(d.new_pos_cost) * min(pos.init_margin_req, margin_req + max(0, inst.funding_rate * sign(d.new_current_qty)));
    x = d.new_pos_cross + d.new_pos_init + d.new_pos_comm + d.new_unrealised_pnl - p
    if pos.cross_margin:
        a = marg.margin_balance - pos.unrealised_pnl + d.new_unrealised_pnl - pos.realised_pnl + d.new_realised_pnl - marg.init_margin - (marg.maint_margin - pos.maint_margin + d.new_maint_margin);
        x += max(0, floor(a / (1 + pos.commission)))
    usd = target_liquidation(d.new_current_qty, new_mark_sign * max(0, new_mark_sign * (d.new_mark_value - x)));
    return round(usd*2)*0.5



def xbt_calc(wallet_balance, contract_qty, entry_price, mark_price, cross_margin, leverage=None):
    instrument = dobj(mark_price=mark_price, funding_rate=0.000169, risk_limit=20000000000, risk_step=10000000000, maint_margin=0.005)
    init_margin_req = 0.01 if cross_margin else 1 / leverage
    position = dobj(commission=0.00075, maint_margin_req=0.005, cross_margin=cross_margin, init_margin_req=init_margin_req, \
                    current_qty=0, current_cost=0, pos_cross=0, realised_cost=0, realised_pnl=0, unrealised_pnl=0, maint_margin=0)
    margin = dobj(margin_balance=int((wallet_balance or 0)*1e8), init_margin=0, maint_margin=0)
    return liquidation_price(instrument, position, margin, contract_qty, entry_price)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wallet-balance', type=float, help='wallet balance in BTC')
    parser.add_argument('--contract-quantity', required=True, type=float, help='contract quantity in USD, negative for shorts')
    parser.add_argument('--entry-price', required=True, type=float, help='entry price in USD')
    parser.add_argument('--mark-price', type=float, help='mark price in USD')
    parser.add_argument('--cross-margin', action='store_true', help='cross margin, alternatively use isolated')
    parser.add_argument('--isolated-margin', action='store_true', help='isolated margin, alternatively use cross margin')
    parser.add_argument('--leverage', type=int, help='leverage to use with isolated margin')
    options = parser.parse_args()
    if not options.cross_margin:
        options.isolated_margin = True
    if options.isolated_margin:
        assert options.leverage
    if options.cross_margin:
        assert options.wallet_balance
    if not options.mark_price:
        options.mark_price = options.entry_price
    print(xbt_calc(options.wallet_balance, options.contract_quantity, options.entry_price, options.mark_price, options.cross_margin, options.leverage))