
import numpy
import sqlite3

class TokenDB:
    volume_stepsize = 5

    def __init__(self, filename):
        self.filename = filename
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS token (ticks_next_tick_aggregate_id INTEGER, volume_next_tick_aggregate_id INTEGER, volume_remaining_volume REAL, volume_remaining_price_high REAL, volume_remaining_price_low REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS trade_data_ticks (timestamp INTEGER, tick_aggregate_id INTEGER, price REAL, volume REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS trade_data_volume (timestamp INTEGER, price_high REAL, price_low REAL)')
        conn.commit()
        conn.close()

    def get_token_state(self):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        c.execute('SELECT ticks_next_tick_aggregate_id, volume_next_tick_aggregate_id, volume_remaining_volume, volume_remaining_price_high, volume_remaining_price_low FROM token')
        try:
            ticks_next_tick_aggregate_id, volume_next_tick_aggregate_id, volume_remaining_volume, volume_remaining_price_high, volume_remaining_price_low = c.fetchall()[0]
        except:
            return None
        conn.close()
        return {'ticks_next_tick_aggregate_id': ticks_next_tick_aggregate_id,
                'volume_next_tick_aggregate_id': volume_next_tick_aggregate_id,
                'volume_remaining_volume': volume_remaining_volume,
                'volume_remaining_price_high': volume_remaining_price_high,
                'volume_remaining_price_low': volume_remaining_price_low}

    def init_token(self, first_tick_aggregate_id = 0):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        c.execute('REPLACE INTO token (ticks_next_tick_aggregate_id, volume_next_tick_aggregate_id, volume_remaining_volume, volume_remaining_price_high, volume_remaining_price_low) VALUES (?, ?, ?, ?, ?)', (first_tick_aggregate_id, first_tick_aggregate_id, 0, 0, 0))
        conn.commit()
        conn.close()

    def append_ticks(self, ticks):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()

        # Get last tick_aggregate_id
        last_tick_aggregate_id = 0
        c.execute('SELECT tick_aggregate_id FROM trade_data_ticks ORDER BY tick_aggregate_id DESC LIMIT 1')
        prices = c.fetchall()
        if prices:
            last_tick_aggregate_id = prices[0][0]

        # Insert ticks
        for tick in ticks:
            if tick['tick_aggregate_id'] <= last_tick_aggregate_id:
                continue
            c.execute("INSERT INTO trade_data_ticks VALUES (?,?,?,?)", (tick['timestamp'],
                                                                        tick['tick_aggregate_id'],
                                                                        tick['price'],
                                                                        tick['volume']))

        next_tick_aggregate_id = ticks[-1]['tick_aggregate_id'] + 1
        c.execute('UPDATE token SET ticks_next_tick_aggregate_id=?', (next_tick_aggregate_id,))
        conn.commit()
        conn.close()

    def append_trade_data_volume(self, volume_data, next_tick_aggregate_id, volume_remaining, price_high, price_low):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        for idx, timestamp in enumerate(volume_data['timestamp']):
            c.execute("INSERT INTO trade_data_volume VALUES (?,?,?)", (timestamp,
                                                                       volume_data['price_high'][idx],
                                                                       volume_data['price_low'][idx]))
        next_tick_aggregate_id += 1
        c.execute('UPDATE token SET volume_next_tick_aggregate_id=?, volume_remaining_volume=?, volume_remaining_price_high=?, volume_remaining_price_low=?', (next_tick_aggregate_id, volume_remaining, price_high, price_low))
        conn.commit()
        conn.close()

    def get_ticks(self, timestamp_start = None, tick_aggregate_id = None, limit = 1):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        if timestamp_start is not None:
            c.execute('SELECT timestamp, tick_aggregate_id, price, volume FROM trade_data_ticks WHERE timestamp>=? LIMIT ?', (timestamp_start, limit))
        elif tick_aggregate_id is not None:
            c.execute('SELECT timestamp, tick_aggregate_id, price, volume FROM trade_data_ticks WHERE tick_aggregate_id>=? LIMIT ?', (tick_aggregate_id, limit))
        else:
            return None
        ticks_raw = c.fetchall()

        if not ticks_raw:
            return None
        ticks = numpy.array(ticks_raw)
        return {'timestamp': ticks[:,0],
                'tick_aggregate_id': ticks[:,1],
                'price': ticks[:,2],
                'volume': ticks[:,3]}

    def aggregate(self, prices, aggregate_rows):
        if aggregate_rows == 1:
            return prices
        
        length = prices.shape[0] // aggregate_rows
        new_prices = numpy.zeros((length, 3))
        for i in range(length):
            sl = prices[i*aggregate_rows:(i+1)*aggregate_rows]
            ts = sl[-1,0]
            low = sl[0:aggregate_rows,1].min()
            high = sl[0:aggregate_rows,2].max()
            new_prices[i,0] = ts
            new_prices[i,1] = low
            new_prices[i,2] = high
        return new_prices

    def get_prices_volume(self, limit = 50, rowid_start = None, aggregate_rows = 1):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        if rowid_start:
            c.execute('SELECT timestamp, price_low, price_high from trade_data_volume WHERE rowid >= ? LIMIT ?;', (rowid_start, limit))
        else:
            c.execute('SELECT timestamp, price_low, price_high from trade_data_volume WHERE rowid > (SELECT rowid FROM trade_data_volume ORDER BY rowid DESC LIMIT 1) - ?', (limit,))
        prices = c.fetchall()
        if not prices:
            return None

        prices = self.aggregate(numpy.array(prices), aggregate_rows)

        return {'timestamp': prices[:,0],
                'low': prices[:,1],
                'high': prices[:,2],
                'volume_stepsize': self.volume_stepsize}
