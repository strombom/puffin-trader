
if False:
    print("NOT RUNNING make_symbols.py")
    quit()

import os
import sys
import requests

symbols = ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "BTTUSDT", "CHZUSDT", "DOGEUSDT", "EOSUSDT", "ETHUSDT", "ETCUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "THETAUSDT", "XLMUSDT", "XRPUSDT"]
symbols = ["BCHUSDT", "THETAUSDT"]

request = requests.get('https://api.bybit.com/v2/public/symbols')

bybit_symbols = request.json()

working_directory = "" # sys.argv[1]
template_file = open(os.path.join(working_directory, "Scripts/Symbols.template"), "r")
symbols_h_str = template_file.read()
template_file.close()

symbols_h_str += '\nconstexpr const auto symbols = std::array{\n'

symbol_idx = 0
for symbol in bybit_symbols['result']:
    if symbol['name'] in symbols:
        name = symbol['name']
        tick_size = symbol['price_filter']['tick_size']
        taker_fee = symbol['taker_fee']
        maker_fee = symbol['maker_fee']
        lot_size = symbol['lot_size_filter']['qty_step']
        min_qty = symbol['lot_size_filter']['min_trading_qty']
        max_qty = symbol['lot_size_filter']['max_trading_qty']

        symbols_h_str += f"    Symbol{{ {symbol_idx}, \"{name}\", {tick_size}, {taker_fee}, {maker_fee}, {lot_size}, {min_qty}, {max_qty} }},\n"
        symbol_idx += 1

symbols_h_str += '};\n'
symbols_h_file = open(os.path.join("./", "Symbols.h"), "w")
symbols_h_file.write(symbols_h_str)
symbols_h_file.close()

print("make_symbols.py done")
