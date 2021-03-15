from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from pyqtgraph import mkPen

from Common.Misc import PositionDirection, Regime
from slopes import Slopes

colors = ['r', 'g', 'y', 'b', ]


class Plotter:
    def __init__(self, n_spectrums):
        self.prices = {'x': [], 'y': []}
        self.spectrums = []

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Puffin')
        self.win.resize(1200, 1000)
        self.win.setBackground((0x34, 0x38, 0x37))

        # Price plot
        self.plt_price = self.win.addPlot(row=0, col=0, rowspan=2)
        self.plt_price.showGrid(x=True, y=True, alpha=0.3)
        self.plt_price.setFixedHeight(self.win.height() * 0.25)
        ax_x_price = self.plt_price.getAxis('bottom')
        ax_x_price.setStyle(showValues=False)
        ax_y_price = self.plt_price.getAxis('left')
        ax_y_price.setWidth(50)

        # Spectrums
        for idx in range(n_spectrums):
            def addSpectrum(row, col, set_x_link=False):
                array = np.empty((200, 30))
                img = pg.ImageItem(array)
                plot = self.win.addPlot(row=idx * 2 + 2, col=0)
                if set_x_link:
                    plot.setXLink(self.plt_price)
                plot.showGrid(x=False, y=False, alpha=0.3)
                plot.addItem(img)
                plot.hideAxis('bottom')
                plot.hideAxis('left')
                return array

            self.spectrums.append(addSpectrum(row=idx * 2 + 2, col=0))

            img_vol = pg.ImageItem(np.empty((200, 30)))
            plot_vol = self.win.addPlot(row=idx * 2 + 2, col=0)
            plot_vol.setXLink(self.plt_price)
            plot_vol.showGrid(x=False, y=False, alpha=0.3)
            plot_vol.addItem(img_vol)
            plot_vol.hideAxis('bottom')
            plot_vol.hideAxis('left')

            img_vol_live = pg.ImageItem(np.empty((20, 30)))
            plot_vol_live = self.win.addPlot(row=idx * 2 + 2, col=1)
            plot_vol_live.showGrid(x=False, y=False, alpha=0.3)
            plot_vol_live.addItem(img_vol_live)
            plot_vol_live.hideAxis('bottom')
            plot_vol_live.hideAxis('left')
            plot_vol_live.setFixedWidth(self.win.width() * 0.2)

            img_dir = pg.ImageItem(np.empty((200, 30)))
            plot_dir = self.win.addPlot(row=idx * 2 + 3, col=0)
            plot_dir.setXLink(self.plt_price)
            plot_dir.showGrid(x=False, y=False, alpha=0.3)
            plot_dir.addItem(img_dir)
            plot_dir.hideAxis('bottom')
            plot_dir.hideAxis('left')

            img_dir_live = pg.ImageItem(np.empty((20, 30)))
            plot_dir_live = self.win.addPlot(row=idx * 2 + 3, col=1)
            plot_dir_live.showGrid(x=False, y=False, alpha=0.3)
            plot_dir_live.addItem(img_dir_live)
            plot_dir_live.hideAxis('bottom')
            plot_dir_live.hideAxis('left')
            plot_dir_live.setFixedWidth(self.win.width() * 0.2)

            self.spectrums.append((img_vol, img_dir))

        # Indicator plot
        """
        self.plt_indicators = self.win.addPlot(row=1, col=0)
        self.plt_indicators.showGrid(x=True, y=True, alpha=0.3)
        self.plt_indicators.setXLink(self.plt_price)
        ax0_x_indicators = self.plt_indicators.getAxis('bottom')
        ax0_x_indicators.setStyle(showValues=False)
        ax0_y_indicators = self.plt_indicators.getAxis('left')
        ax0_y_indicators.setWidth(50)
        """

        # Value plot
        # self.plt_value = self.win.addPlot(row=1, col=0)
        # self.plt_value.showGrid(x=True, y=True, alpha=0.3)
        # self.plt_value.setXLink(self.plt_price)
        # ax_y_value = self.plt_value.getAxis('left')
        # ax_y_value.setWidth(50)

        # self.win.ci.layout.setSpacing(0)
        # self.win.ci.layout.setContentsMargins(0, 0, 0, 0)
        # self.win.ci.layout.setRowStretchFactor(0, 6)
        # self.win.ci.layout.setRowStretchFactor(1, 2)
        # self.win.ci.layout.setRowStretchFactor(2, 1)

        self.mouse_move_signal = pg.SignalProxy(self.plt_price.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_move)

        """
        self.annotation_fill = {'positive': pg.mkBrush(color=(0x15, 0xb0, 0xa1, 40)),
                                'negative': pg.mkBrush(color=(0xe5, 0x00, 0x00, 40))}

        crosshair_pen = pg.mkPen({'color': (0xff, 0xff, 0xff, 25), 'width': 2})
        self.v_line_price = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.v_line_indicators = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.v_line_value = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.plt_price.addItem(self.v_line_price, ignoreBounds=True)
        #self.plt_indicators.addItem(self.v_line_indicators, ignoreBounds=True)
        self.plt_value.addItem(self.v_line_value, ignoreBounds=True)

        self.datetime_marker = pg.TextItem(f'DT', anchor=(0.5, 1), fill=(0x81, 0x81, 0x81))
        self.plt_price.addItem(self.datetime_marker)

        self.plt_indicator_annotations = {'volatility': pg.TextItem(f'V', anchor=(0.5, 0), fill=(0x00, 0x35, 0x5b)),
                                          'length': pg.TextItem(f'L', anchor=(0.5, 0), fill=(0x02, 0xab, 0x2e)),
                                          'angle': pg.TextItem(f'A', anchor=(0.5, 0), fill=(0x84, 0x00, 0x00))}
        #for name, annotation in self.plt_indicator_annotations.items():
        #    self.plt_indicators.addItem(annotation)
        """

    """
    def append_event(self, event_direction: PositionDirection, event_idx: int, event_price: float) -> None:
        self.events[event_direction]['x'].append(event_idx)
        self.events[event_direction]['y'].append(event_price)

    def append_timestamp(self, event_idx: int,  timestamp: str) -> None:
        self.timestamps['x'].append(event_idx)
        self.timestamps['y'].append(timestamp)

    def append_angle(self, angle_idx: int, angle: float) -> None:
        self.angles['x'].append(angle_idx)
        self.angles['y'].append(angle)

    def append_value(self, value_idx: int, value_price: float) -> None:
        self.values['x'].append(value_idx)
        self.values['y'].append(value_price)

    def append_threshold(self, threshold_idx: int, threshold: float) -> None:
        self.thresholds['x'].append(threshold_idx)
        self.thresholds['y'].append(threshold)

    def append_volatility(self, volatility_idx: int, volatility: float) -> None:
        self.volatilities['x'].append(volatility_idx)
        self.volatilities['y'].append(volatility)

    def append_annotation(self, x: int, y: float, direction: PositionDirection, profit: float) -> None:
        if profit >= 0:
            fill = self.annotation_fill['positive']
        else:
            fill = self.annotation_fill['negative']
        annotation = pg.TextItem(f'{profit * 100:.2f}', anchor=(0.5, 1), fill=fill)
        annotation.setPos(x, y + 100)
        self.plt_price.addItem(annotation)
        arrow = pg.ArrowItem(angle=270)
        arrow.setPos(x, y + 30)
        self.plt_price.addItem(arrow)
    
    def regime_change(self, x: int, mark_price: float, regime: Regime):
        if regime == Regime.trend:
            text = 'Trend'
            brush = pg.mkBrush(color=(0xc7, 0xfb, 0xb5, 30))
        else:
            text = 'Chop'
            brush = pg.mkBrush(color=(0xff, 0xb0, 0x7c, 30))

        annotation = pg.TextItem(f'{text}', anchor=(0.5, 0), fill=brush)
        annotation.setPos(x, mark_price - 100)
        self.plt_price.addItem(annotation)
        arrow = pg.ArrowItem(angle=90)
        arrow.setPos(x, mark_price - 30)
        self.plt_price.addItem(arrow)
    """

    def mouse_move(self, event) -> None:
        pos = event[0]
        mouse_point = self.plt_price.vb.mapSceneToView(pos)
        # if self.plt_price.sceneBoundingRect().contains(pos):
        x, y = mouse_point.x(), mouse_point.y()
        x = int(x + 0.5)

        """
        self.v_line_price.setPos(x)
        self.v_line_indicators.setPos(x)
        self.v_line_value.setPos(x)

        if 0 <= x < len(self.timestamps['x']):
            timestamp = self.timestamps['y'][x] / 1000
            timestamp = datetime.fromtimestamp(timestamp)
            self.datetime_marker.setPos(x, self.plt_price.getAxis('left').range[0] + 200)
            self.datetime_marker.setText(text=timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        for idx, (name, annotation) in enumerate(self.plt_indicator_annotations.items()):
            annotation.setPos(x + (idx - 1) * 7, -0.7)
        """

    """
    def add_indicator(self, data: np.ndarray, label: str):
        x = np.arange(data.shape[0])

        runner_price_pen = pg.mkPen(size=2, brush=pg.mkBrush(0xff, 0xff, 0x14, 200))
        runner_price_line = self.plt_price.plot(x, data, pen=runner_price_pen)
    """

    def show(self) -> None:
        self.win.show()
        self.app.exec_()

        """
        y_min, y_max = 1e9, 0
        event_colors = {PositionDirection.long:  pg.mkBrush(0x15, 0xb0, 0x1a, 250),
                        PositionDirection.hedge: pg.mkBrush(0x95, 0xd0, 0xfc, 250),
                        PositionDirection.short: pg.mkBrush(0xe5, 0x00, 0x00, 250)}
        for direction in [PositionDirection.long, PositionDirection.hedge, PositionDirection.short]:
            events = self.events[direction]
            if len(events['y']) == 0:
                continue
            scatter = pg.ScatterPlotItem(size=5, brush=event_colors[direction])
            scatter.addPoints(events['x'], events['y'])
            self.plt_price.addItem(scatter)
            y_min = min(y_min, min(events['y']))
            y_max = max(y_max, max(events['y']))

            runner_price_pen = pg.mkPen(size=2, brush=pg.mkBrush(0xe5, 0xd0, 0x00, 30))
            runner_price_line = self.plt_price.plot(events['x'], events['y'], pen=runner_price_pen)

        scatter_symbol = QtGui.QPainterPath()
        scatter_symbol.addText(0, 0, QtGui.QFont("San Serif", 8), '-')
        scatter_br = scatter_symbol.boundingRect()
        scale = min(1. / scatter_br.width(), 1. / scatter_br.height())
        scatter_tr = QtGui.QTransform()
        scatter_tr.scale(scale, scale)
        scatter_tr.translate(-scatter_br.x() - scatter_br.width() / 2., -scatter_br.y() - scatter_br.height() / 2.)
        scatter_symbol = scatter_tr.map(scatter_symbol)

        scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(0xe5, 0xd0, 0x00, 10), symbol=scatter_symbol)
        scatter.addPoints(self.thresholds['x'], self.thresholds['y'])
        self.plt_price.addItem(scatter)

        self.plt_price.setLimits(yMin=y_min * 0.9, yMax=y_max * 1.1)
        # self.plt.enableAutoRange()

        self.plt_value.plot(self.values['x'], self.values['y'])

        #self.plt_indicators.addLegend()
        #self.plt_indicators.setLabel('left', 'Abc')
        #self.plt_indicators.plot(self.volatilities['x'], self.volatilities['y'], pen='b', name=f'Volatility')
        #self.plt_indicators.plot(self.angles['x'], self.angles['y'], pen='r', name=f'Angle')

        self.win.show()
        self.app.exec_()
        """

