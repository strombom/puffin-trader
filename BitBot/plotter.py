import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from Common.Misc import PositionDirection

colors = ['r', 'g', 'y', 'b', ]


class Plotter:
    def __init__(self):
        self.events = {
            PositionDirection.long:  {'x': [], 'y': []},
            PositionDirection.hedge: {'x': [], 'y': []},
            PositionDirection.short: {'x': [], 'y': []}
        }
        self.angles = {'x': [], 'y': []}
        self.values = {'x': [], 'y': []}
        self.thresholds = {'x': [], 'y': []}

    def append_event(self, event_direction: PositionDirection, event_idx: int, event_price: float):
        self.events[event_direction]['x'].append(event_idx)
        self.events[event_direction]['y'].append(event_price)

    def append_angle(self, angle_idx: int, angle: float):
        self.angles['x'].append(angle_idx)
        self.angles['y'].append(angle)

    def append_value(self, value_idx: int, value_price: float):
        self.values['x'].append(value_idx)
        self.values['y'].append(value_price)

    def append_threshold(self, threshold_idx: int, threshold: float):
        self.thresholds['x'].append(threshold_idx)
        self.thresholds['y'].append(threshold)

    def plot(self):
        app = QtGui.QApplication([])
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('Puffin')
        win.resize(1200, 1000)
        win.setBackground((0x34, 0x38, 0x37))
        img = pg.ImageItem()
        plot = win.addPlot()
        plot.addItem(img)

        y_min, y_max = 1e9, 0

        event_colors = {PositionDirection.long:  pg.mkBrush(0x15, 0xb0, 0x1a, 220),
                        PositionDirection.hedge: pg.mkBrush(0x95, 0xd0, 0xfc, 220),
                        PositionDirection.short: pg.mkBrush(0xe5, 0x00, 0x00, 220)}
        for direction in [PositionDirection.long, PositionDirection.hedge, PositionDirection.short]:
            events = self.events[direction]
            scatter = pg.ScatterPlotItem(size=4, brush=event_colors[direction])
            scatter.addPoints(events['x'], events['y'])
            plot.addItem(scatter)
            y_min = min(y_min, min(events['y']))
            y_max = max(y_max, max(events['y']))

        plot.setLimits(yMin=y_min * 0.95, yMax=y_max * 1.05)
        # plot.enableAutoRange()
        win.show()

        app.exec_()
