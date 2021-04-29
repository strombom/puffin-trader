
from binance.websockets import BinanceSocketManager


class BinanceAccount:
    def __init__(self, binance_client, top_symbols):
        self.mark_prices = {}

        self._client = binance_client
        self._assets = {}
        self._kline_threads = {}

        all_symbols = {}
        exchange_info = self._client.get_exchange_info()
        for symbol in exchange_info['symbols']:
            if symbol['baseAsset'] not in all_symbols:
                all_symbols[symbol['baseAsset']] = []
            all_symbols[symbol['baseAsset']].append(symbol['symbol'])

        trade_symbols = []
        for top_symbol in top_symbols:
            if top_symbol in all_symbols:
                for symbol in all_symbols[top_symbol]:
                    if 'USDT' in symbol:
                        trade_symbols.append(top_symbol)
                        break

        def process_trade_message(data):
            if data['e'] == 'kline':
                self.mark_prices[data['s']] = float(data['k']['c'])

        for symbol in trade_symbols:
            trade_socket_manager = BinanceSocketManager(self._client)
            trade_socket_manager.start_kline_socket(symbol=symbol + "USDT", callback=process_trade_message, interval='1m')
            trade_socket_manager.start()
            self._kline_threads[symbol] = trade_socket_manager

        from time import sleep
        has_all_symbols = False
        while not has_all_symbols:
            has_all_symbols = True
            for symbol in trade_symbols:
                if symbol + "USDT" not in self.mark_prices:
                    has_all_symbols = False
                    sleep(0.2)
                    break
