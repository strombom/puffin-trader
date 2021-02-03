import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui

from Common.Misc import PositionDirection

colors = ['r', 'g', 'y', 'b', ]


class Plotter:
    def __init__(self, slopes):
        self.slopes = slopes
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

        # Add plots
        self.plt = self.win.addPlot(row=0, col=0)
        self.plt.showGrid(x=True, y=True, alpha=0.3)
        ax0 = self.plt.getAxis('bottom')
        ax0.setStyle(showValues=False)
        self.plt_value = self.win.addPlot(row=1, col=0)
        self.plt_value.showGrid(x=True, y=True, alpha=0.3)
        self.plt_value.setXLink(self.plt)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        self.win.ci.layout.setSpacing(0)
        self.win.ci.layout.setContentsMargins(0, 0, 0, 0)

        # Crosshair
        # crosshair_pen = pg.mkPen({'color': (0xff, 0xff, 0xff, 25), 'width': 2})
        # self.v_top_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        # self.v_bot_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        # self.h_line = pg.InfiniteLine(angle=0,  movable=False, pen=crosshair_pen)
        # self.plt.addItem(self.v_top_line, ignoreBounds=True)
        # self.plt_value.addItem(self.v_bot_line, ignoreBounds=True)
        # self.plt.addItem(self.h_line, ignoreBounds=True)

        self.slope_lines = []
        for idx in range(40):
            slope_pen = pg.mkPen({'color': (0x02, 0xbf, 0xfe, 25 + idx * 1), 'width': 2})
            slope_line = self.plt.plot([0, 1], [0, 1], pen=slope_pen)  # pg.ScatterPlotItem(size=4, brush=event_colors[direction])
            self.slope_lines.append(slope_line)

        self.mouse_move_signal = pg.SignalProxy(self.plt.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_move)

        self.annotation_fill = {'positive': pg.mkBrush(color=(0x15, 0xb0, 0xa1, 80)),
                                'negative': pg.mkBrush(color=(0xe5, 0x00, 0x00, 80))}

    def append_event(self, event_direction: PositionDirection, event_idx: int, event_price: float) -> None:
        self.events[event_direction]['x'].append(event_idx)
        self.events[event_direction]['y'].append(event_price)

    def append_angle(self, angle_idx: int, angle: float) -> None:
        self.angles['x'].append(angle_idx)
        self.angles['y'].append(angle)

    def append_value(self, value_idx: int, value_price: float) -> None:
        self.values['x'].append(value_idx)
        self.values['y'].append(value_price)

    def append_threshold(self, threshold_idx: int, threshold: float) -> None:
        self.thresholds['x'].append(threshold_idx)
        self.thresholds['y'].append(threshold)

    def append_annotation(self, x: int, y: float, direction: PositionDirection, profit: float) -> None:
        if profit >= 0:
            fill = self.annotation_fill['positive']
        else:
            fill = self.annotation_fill['negative']
        annotation = pg.TextItem(f'{profit * 100:.2f}', anchor=(0.5, 1), fill=fill)
        annotation.setPos(x, y + 80)
        self.plt.addItem(annotation)
        arrow = pg.ArrowItem(angle=270)
        arrow.setPos(x, y + 30)
        self.plt.addItem(arrow)

    def mouse_move(self, event) -> None:
        pos = event[0]
        mouse_point = self.plt.vb.mapSceneToView(pos)
        if self.plt.sceneBoundingRect().contains(pos):
            x, y = mouse_point.x(), mouse_point.y()

            x = int(x + 0.5)
            if self.slopes.max_slope_length + len(self.slope_lines) <= x < self.slopes.max_slope_length + len(self.slopes):
                for idx in range(len(self.slope_lines)):
                    slope = self.slopes[x - self.slopes.max_slope_length - len(self.slope_lines) + 1 + idx]
                    self.slope_lines[idx].setData(slope.x, slope.y)

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

        # for annotation in self.annotations:
        #     annotation.setParentItem()

        self.plt_value.plot(self.values['x'], self.values['y'])

        self.plt.setLimits(yMin=y_min * 0.9, yMax=y_max * 1.1)
        # self.plt.enableAutoRange()

        self.win.show()
        self.app.exec_()
