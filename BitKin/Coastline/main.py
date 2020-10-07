
import sys
sys.path.append("../Common")

from Coastline import CoastlineTrader, TraderDirection


trader = CoastlineTrader(0.0025, TraderDirection.long)
