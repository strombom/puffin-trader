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

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Puffin')
        self.win.resize(1200, 1000)
        self.win.setBackground((0x34, 0x38, 0x37))
        self.plt = self.win.addPlot()
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.plt.addItem(self.v_line, ignoreBounds=True)
        self.plt.addItem(self.h_line, ignoreBounds=True)
        self.mouse_move_signal = pg.SignalProxy(self.plt.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_move)

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

    def mouse_move(self, event):
        pos = event[0]
        mouse_point = self.plt.vb.mapSceneToView(pos)
        if self.plt.sceneBoundingRect().contains(pos):
            x, y = mouse_point.x(), mouse_point.y()
            self.v_line.setPos(x)
            self.h_line.setPos(y)

    def plot(self) -> None:
        y_min, y_max = 1e9, 0
        event_colors = {PositionDirection.long:  pg.mkBrush(0x15, 0xb0, 0x1a, 220),
                        PositionDirection.hedge: pg.mkBrush(0x95, 0xd0, 0xfc, 220),
                        PositionDirection.short: pg.mkBrush(0xe5, 0x00, 0x00, 220)}
        for direction in [PositionDirection.long, PositionDirection.hedge, PositionDirection.short]:
            events = self.events[direction]
            scatter = pg.ScatterPlotItem(size=4, brush=event_colors[direction])
            scatter.addPoints(events['x'], events['y'])
            self.plt.addItem(scatter)
            y_min = min(y_min, min(events['y']))
            y_max = max(y_max, max(events['y']))

        self.plt.setLimits(yMin=y_min * 0.95, yMax=y_max * 1.05)
        # self.plt.enableAutoRange()

        self.win.show()
        self.app.exec_()
