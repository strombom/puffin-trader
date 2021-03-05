
import pickle
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt5 import QtGui

class RunnerPlotter:
    def __init__(self, runner_data: pd.DataFrame, tick_data: pd.DataFrame):
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
        ax0_x_indicators.setStyle(showValues=False)
        ax0_y_indicators = self.plt_indicators.getAxis('left')
        ax0_y_indicators.setWidth(50)

        self.win.ci.layout.setSpacing(0)
        self.win.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.win.ci.layout.setRowStretchFactor(0, 3)
        self.win.ci.layout.setRowStretchFactor(1, 2)

        self.mouse_move_signal = pg.SignalProxy(self.plt_price.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_move)

        crosshair_pen = pg.mkPen({'color': (0xff, 0xff, 0xff, 25), 'width': 2})
        self.v_line_price = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.v_line_indicators = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.plt_price.addItem(self.v_line_price, ignoreBounds=True)
        self.plt_indicators.addItem(self.v_line_indicators, ignoreBounds=True)

        # tick_pen = pg.mkPen({'color': (0x02, 0xbf, 0xfe, 50), 'width': 2})
        # tick_x, tick_y = tick_data[['timestamp']].to_numpy().squeeze(), tick_data[['price']].to_numpy().squeeze()
        # tick_line = self.plt_price.plot(tick_x, tick_y, pen=tick_pen)

        runner_timestamp = runner_data[['timestamp']].to_numpy().squeeze()
        runner_price = runner_data[['price']].to_numpy().squeeze()
        runner_price_max = runner_data[['price_max']].to_numpy().squeeze()
        runner_price_min = runner_data[['price_min']].to_numpy().squeeze()
        runner_x = np.arange(runner_timestamp.shape[0])

        scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(0xe5, 0xd0, 0x00, 100))
        scatter.addPoints(runner_x, runner_price)
        self.plt_price.addItem(scatter)

        runner_price_pen = pg.mkPen(size=2, brush=pg.mkBrush(0xe5, 0xd0, 0x00, 100))
        runner_price_line = self.plt_price.plot(runner_x, runner_price, pen=runner_price_pen)

        price_max_x = np.arange(runner_timestamp.shape[0]).repeat(2, axis=0)[1:-1]
        runner_price_max2 = runner_price_max.repeat(2, axis=0)[2:]
        runner_price_max_pen = pg.mkPen({'color': (0x96, 0xf9, 0x7b, 40), 'width': 2})
        runner_price_max_line = self.plt_price.plot(price_max_x, runner_price_max2, pen=runner_price_max_pen)

        price_min_x = np.arange(runner_timestamp.shape[0]).repeat(2, axis=0)[1:-1]
        runner_price_min2 = runner_price_min.repeat(2, axis=0)[2:]
        runner_price_min_pen = pg.mkPen({'color': (0xe5, 0x00, 0x00, 40), 'width': 2})
        runner_price_min_line = self.plt_price.plot(price_min_x, runner_price_min2, pen=runner_price_min_pen)

        """
        price_max_lines = []
        price_min_lines = []
        line_max_pen = pg.mkPen({'color': (0x96, 0xf9, 0x7b, 100), 'width': 2})
        line_min_pen = pg.mkPen({'color': (0xe5, 0x00, 0x00, 100), 'width': 2})
        for idx in range(1, runner_timestamp.shape[0]):
            pass
            # price_max_lines.append(self.plt_price.plot(runner_timestamp[idx-1:idx+1], [runner_price_max[idx], runner_price_max[idx]], pen=line_max_pen))
            price_max_lines.append(self.plt_price.plot([idx, idx], [runner_price_max[idx], runner_price[idx]], pen=line_max_pen))
            #price_min_lines.append(self.plt_price.plot(runner_timestamp[idx-1:idx+1], [runner_price_min[idx], runner_price_min[idx]], pen=line_min_pen))
            price_min_lines.append(self.plt_price.plot([idx, idx], [runner_price_min[idx], runner_price[idx]], pen=line_min_pen))
        """

        """
        self.slope_lines = []
        for idx in range(40):
            slope_pen = pg.mkPen({'color': (0x02, 0xbf, 0xfe, 25 + idx * 1), 'width': 2})
            slope_line = self.plt_price.plot([0, 1], [0, 1], pen=slope_pen)  # pg.ScatterPlotItem(size=4, brush=event_colors[direction])
            self.slope_lines.append(slope_line)

        # self.plt_indicator_annotations = {'volatility': pg.TextItem(f'V', anchor=(0.5, 0), fill=(0x00, 0x35, 0x5b)),
        #                                   'length': pg.TextItem(f'L', anchor=(0.5, 0), fill=(0x02, 0xab, 0x2e)),
        #                                   'angle': pg.TextItem(f'A', anchor=(0.5, 0), fill=(0x84, 0x00, 0x00))}
        # for name, annotation in self.plt_indicator_annotations.items():
        #     self.plt_indicators.addItem(annotation)

        self.vol_diff = self.plt_indicators.plot([0, 1], [0, 1], pen='r', name=f'Volatility diff')
        """

    def plot(self) -> None:
        y_min, y_max = 1e9, 0
        event_color = pg.mkBrush(0x15, 0xb0, 0x1a, 250)
        scatter = pg.ScatterPlotItem(size=5, brush=event_color)
        scatter.addPoints(self.events['x'], self.events['y'])
        self.plt_price.addItem(scatter)

        self.plt_indicators.addLegend()
        self.plt_indicators.setLabel('left', 'Indicators')
        # self.plt_indicators.plot(self.volatilities['x'], self.volatilities['y'], pen='b', name=f'Volatility')
        # self.plt_indicators.plot(self.slope_lengths['x'], self.slope_lengths['y'], pen='g', name=f'Length')
        # self.plt_indicators.plot(self.angles['x'], self.angles['y'], pen='r', name=f'Angle')
        #self.plt_indicators.plot([0], [0], pen='r', name=f'Volatility diff')

        # self.plt_indicators.setLimits(yMin=0, yMax=0.02)
        self.plt_indicators.setYRange(min=-1, max=1)

        self.win.show()
        self.app.exec_()

    def mouse_move(self, event) -> None:
        pos = event[0]
        mouse_point = self.plt_price.vb.mapSceneToView(pos)
        # if self.plt_price.sceneBoundingRect().contains(pos):
        pos_x, pos_y = mouse_point.x(), mouse_point.y()
        pos_x = int(pos_x + 0.5)

        self.v_line_price.setPos(pos_x)
        self.v_line_indicators.setPos(pos_x)


if __name__ == '__main__':
    runner_data = pd.read_csv('../tmp/binance_runner.csv')
    tick_data = pd.read_csv('../tmp/binance_runner_ticks.csv')
    # tick_data = None
    plotter = RunnerPlotter(runner_data=runner_data, tick_data=tick_data)
    plotter.plot()
