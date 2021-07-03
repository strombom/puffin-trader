
import pickle
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtGui
from slopes import Slopes


class SlopesPlotter:
    def __init__(self, slopes: Slopes):
        self.slopes = slopes

        self.events = {'x': [], 'y': []}

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Puffin')
        self.win.resize(1200, 1000)
        self.win.setBackground((0x34, 0x38, 0x37))

        # Price plot
        self.plt_price = self.win.addPlot(row=0, col=0)
        self.plt_price.showGrid(x=True, y=True, alpha=0.3)
        ax_x_price = self.plt_price.getAxis('bottom')
        ax_x_price.setStyle(showValues=False)
        ax_y_price = self.plt_price.getAxis('left')
        ax_y_price.setWidth(50)

        # Indicator plot
        self.plt_indicators = self.win.addPlot(row=1, col=0)
        self.plt_indicators.showGrid(x=True, y=True, alpha=0.3)
        self.plt_indicators.setXLink(self.plt_price)
        ax0_x_indicators = self.plt_indicators.getAxis('bottom')
        # ax0_x_indicators.setStyle(showValues=False)
        ax0_y_indicators = self.plt_indicators.getAxis('left')
        ax0_y_indicators.setWidth(50)

        self.win.ci.layout.setSpacing(0)
        self.win.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.win.ci.layout.setRowStretchFactor(0, 3)
        self.win.ci.layout.setRowStretchFactor(1, 2)

        self.slope_lines = []
        for idx in range(40):
            slope_pen = pg.mkPen({'color': (0x02, 0xbf, 0xfe, 25 + idx * 1), 'width': 2})
            slope_line = self.plt_price.plot([0, 1], [0, 1], pen=slope_pen)  # pg.ScatterPlotItem(size=4, brush=event_colors[direction])
            self.slope_lines.append(slope_line)

        self.mouse_move_signal = pg.SignalProxy(self.plt_price.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_move)

        self.annotation_fill = {'positive': pg.mkBrush(color=(0x15, 0xb0, 0xa1, 40)),
                                'negative': pg.mkBrush(color=(0xe5, 0x00, 0x00, 40))}

        crosshair_pen = pg.mkPen({'color': (0xff, 0xff, 0xff, 25), 'width': 2})
        self.v_line_price = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.v_line_indicators = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.plt_price.addItem(self.v_line_price, ignoreBounds=True)
        self.plt_indicators.addItem(self.v_line_indicators, ignoreBounds=True)

        # self.plt_indicator_annotations = {'volatility': pg.TextItem(f'V', anchor=(0.5, 0), fill=(0x00, 0x35, 0x5b)),
        #                                   'length': pg.TextItem(f'L', anchor=(0.5, 0), fill=(0x02, 0xab, 0x2e)),
        #                                   'angle': pg.TextItem(f'A', anchor=(0.5, 0), fill=(0x84, 0x00, 0x00))}
        # for name, annotation in self.plt_indicator_annotations.items():
        #     self.plt_indicators.addItem(annotation)

        self.vol_diff = self.plt_indicators.plot([0, 1], [0, 1], pen='r', name=f'Volatility diff')

    def append_event(self, event_idx: int, event_price: float) -> None:
        self.events['x'].append(event_idx)
        self.events['y'].append(event_price)

    def mouse_move(self, event) -> None:
        pos = event[0]
        mouse_point = self.plt_price.vb.mapSceneToView(pos)
        # if self.plt_price.sceneBoundingRect().contains(pos):
        pos_x, pos_y = mouse_point.x(), mouse_point.y()
        pos_x = int(pos_x + 0.5)

        self.v_line_price.setPos(pos_x)
        self.v_line_indicators.setPos(pos_x)

        if self.slopes.max_slope_length + len(self.slope_lines) <= pos_x < self.slopes.max_slope_length + len(self.slopes):
            for idx in range(len(self.slope_lines)):
                slope = self.slopes[pos_x - self.slopes.max_slope_length - len(self.slope_lines) + 1 + idx]
                self.slope_lines[idx].setData(slope.x, slope.y)

        if self.slopes.max_slope_length <= pos_x < self.slopes.max_slope_length + len(self.slopes):
            slope = self.slopes[pos_x - self.slopes.max_slope_length]
            vol_x = np.arange(pos_x - len(slope.volatilities) - self.slopes.min_slope_length, pos_x - self.slopes.min_slope_length)
            self.vol_diff.setData(vol_x, slope.volatilities)

        # for idx, (name, annotation) in enumerate(self.plt_indicator_annotations.items()):
        #    annotation.setPos(x + (idx - 1) * 7, -0.7)

        # idx = x - Slopes.max_slope_length
        # if 0 <= idx < len(self.volatilities['y']):
        #    self.plt_indicator_annotations['volatility'].setText(text=f"{self.volatilities['y'][idx]:.2f}")
        #    self.plt_indicator_annotations['length'].setText(text=f"{self.slope_lengths['y'][idx]:.2f}")
        #    self.plt_indicator_annotations['angle'].setText(text=f"{self.angles['y'][idx]:.2f}")

    def plot(self) -> None:
        y_min, y_max = 1e9, 0
        event_color = pg.mkBrush(0x15, 0xb0, 0x1a, 250)
        scatter = pg.ScatterPlotItem(size=5, brush=event_color)
        scatter.addPoints(self.events['x'], self.events['y'])
        self.plt_price.addItem(scatter)
        y_min = min(y_min, min(self.events['y']))
        y_max = max(y_max, max(self.events['y']))

        # scatter_symbol = QtGui.QPainterPath()
        # scatter_symbol.addText(0, 0, QtGui.QFont("San Serif", 8), '-')
        # scatter_br = scatter_symbol.boundingRect()
        # scale = min(1. / scatter_br.width(), 1. / scatter_br.height())
        # scatter_tr = QtGui.QTransform()
        # scatter_tr.scale(scale, scale)
        # scatter_tr.translate(-scatter_br.x() - scatter_br.width() / 2., -scatter_br.y() - scatter_br.height() / 2.)
        # scatter_symbol = scatter_tr.map(scatter_symbol)

        # scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(0xe5, 0xd0, 0x00, 10), symbol=scatter_symbol)
        # scatter.addPoints(self.thresholds['x'], self.thresholds['y'])
        # self.plt_price.addItem(scatter)

        self.plt_price.setLimits(yMin=y_min * 0.9, yMax=y_max * 1.1)
        # self.plt.enableAutoRange()

        self.plt_indicators.addLegend()
        self.plt_indicators.setLabel('left', 'Abc')
        # self.plt_indicators.plot(self.volatilities['x'], self.volatilities['y'], pen='b', name=f'Volatility')
        # self.plt_indicators.plot(self.slope_lengths['x'], self.slope_lengths['y'], pen='g', name=f'Length')
        # self.plt_indicators.plot(self.angles['x'], self.angles['y'], pen='r', name=f'Angle')
        self.plt_indicators.plot([0], [0], pen='r', name=f'Volatility diff')

        #self.plt_indicators.setLimits(yMin=0, yMax=0.02)
        self.plt_indicators.setYRange(min=-1, max=1)

        self.win.show()
        self.app.exec_()


if __name__ == '__main__':
    with open(f"cache/intrinsic_time_runner.pickle", 'rb') as f:
        delta, runner = pickle.load(f)

    runner.ie_prices = np.array(runner.ie_prices)[0:2000]
    slopes = Slopes(runner.ie_prices, use_cache=False)
    x = np.arange(runner.ie_prices.shape[0])

    plotter = SlopesPlotter(slopes=slopes)
    for idx, ie_price in enumerate(runner.ie_prices):
        plotter.append_event(idx, ie_price)

    plotter.plot()
