""" python threading """

import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt6 import QtCore as qtc
from PyQt6 import QtWidgets as qtw

from qcore.helpers.logger import logger
from qcore.variables.datasets import Dataset
from qcore.variables.sweeps import Sweep


class PlotterInitializationError(Exception):
    """ """


class PlotSpec:
    """ """

    def __init__(self, dataset: Dataset) -> None:
        """ """
        self.plot_item = pg.PlotItem()
        self.fit_label = None  # will be set by Plotter

        # determine plot type
        supported_plot_types = ("scatter", "line", "image")
        self.plot_type = "scatter"  # by default
        if "plot_type" in dataset.plot_args:
            plot_type = dataset.plot_args["plot_type"]
            if plot_type not in supported_plot_types:
                msg = f"Unsupported {plot_type = }, {supported_plot_types = }."
                logger.error(msg)
                raise PlotterInitializationError(msg)
            self.plot_type = plot_type

        if self.plot_type in ("scatter", "line"):
            self.plot_legend = self.plot_item.addLegend(offset=(-1, 1))

        # determine the number of data items per plot
        shape = dataset.shape
        data_dim = len(shape) - 1  # discard 1 averaging dimension "N"
        self.num_data_items = 1  # default number of data items in one plot item
        if data_dim == 2 and not self.plot_type == "image":
            self.num_data_items = shape[-2]
        elif data_dim == 1 and self.plot_type == "image":
            msg = f"Invalid dataset {shape = } dimensions for '{self.plot_type}' plot."
            logger.error(msg)
            raise PlotterInitializationError(msg)

        if self.num_data_items > Plotter.MAX_DATA_ITEMS:
            msg = f"Maximum {Plotter.MAX_DATA_ITEMS} data items per plot item allowed."
            logger.error(msg)
            raise PlotterInitializationError(msg)

        # determine whether or not to plot errorbars, default = True
        self.plot_err = True
        if self.plot_type == "image":
            self.plot_err = False
        elif "plot_err" in dataset.plot_args:
            self.plot_err = dataset.plot_args["plot_err"]

        # initialize pyqtgraph graphics objects and add them to the plot item
        self.plot_data_items = []
        self.plot_err_items = []
        self.plot_fit_items = []
        for i in range(self.num_data_items):
            color = (i, self.num_data_items)
            if self.plot_type == "scatter":
                size = Plotter.SCATTER_DOT_SIZE
                plot_data_item = pg.ScatterPlotItem(pen=None, size=size, brush=color)
            elif self.plot_type == "line":
                plot_data_item = pg.PlotCurveItem(pen=color)
            elif self.plot_type == "image":
                plot_data_item = pg.ImageItem()
                cmap = dataset.plot_args.get("cmap", "viridis")
                self.cbar = self.plot_item.addColorBar(
                    plot_data_item, colorMap=cmap, interactive=False
                )
            self.plot_data_items.append(plot_data_item)
            self.plot_item.addItem(plot_data_item)

            if self.plot_err:
                plot_err_item = pg.ErrorBarItem(pen={"color": color, "width": 3})
                self.plot_err_items.append(plot_err_item)
                self.plot_item.addItem(plot_err_item)

            if dataset.fitfn is not None:
                pen = pg.mkPen(color=color, style=qtc.Qt.PenStyle.DashLine)
                plot_fit_item = pg.PlotCurveItem(pen=pen)
                self.plot_fit_items.append(plot_fit_item)
                self.plot_item.addItem(plot_fit_item)

        # set legends
        if self.num_data_items == 1 and not self.plot_type == "image":
            self.plot_legend.addItem(self.plot_data_items[0], f"{dataset.name}_avg")
            if dataset.fitfn is not None:
                self.plot_legend.addItem(self.plot_fit_items[0], f"{dataset.name}_fit")
        elif not self.plot_type == "image":
            y = list(dataset.sweep_data.values())[-2]
            for i in range(self.num_data_items):
                to_round = (float, np.floating)
                ytxt = f"{y[i]:.5f}" if isinstance(y[i], to_round) else f"{y[i]}"
                self.plot_legend.addItem(self.plot_data_items[i], ytxt)

        # add a crosshair for mouse interaction
        cx = pg.InfiniteLine(angle=90, movable=False)
        cy = pg.InfiniteLine(angle=0, movable=False)
        self.plot_item.addItem(cx, ignoreBounds=True)
        self.plot_item.addItem(cy, ignoreBounds=True)
        self.crosshair = (cx, cy)

        # set non-changing aspects of the plot item e.g. axis labels, title, grid
        axes = dataset.axes.copy()
        xlabel = dataset.plot_args.get("xlabel", "")
        ylabel = dataset.plot_args.get("ylabel", "")
        title = dataset.plot_args.get("title", "")
        if not xlabel:
            if axes and isinstance(axes[-1], Sweep):
                xaxis = axes[-1]
                xlabel = f"{xaxis.name} ({xaxis.units})"
        if not ylabel:
            if data_dim == 2 and self.plot_type == "image":
                yaxis = axes[-2]
                if isinstance(yaxis, Sweep):
                    ylabel = f"{yaxis.name} ({yaxis.units})"
            else:
                ylabel = f"{dataset.name} ({dataset.units})"
        if not title:
            if data_dim == 2 and self.plot_type == "image":
                title = f"{dataset.name} ({dataset.units}) vs [{ylabel}, {xlabel}]"
            else:
                title = f"{dataset.name} ({dataset.units}) vs {xlabel}"

        self.plot_item.setLabels(bottom=xlabel, left=ylabel, title=title)
        self.plot_item.showGrid(x=True, y=True, alpha=0.5)
        self.plot_item.setMenuEnabled(False)


class PlotWidget(pg.GraphicsLayoutWidget):
    """ """

    def __init__(self, filename, *args, **kwargs):
        """ """
        super().__init__(*args, **kwargs)
        self.filename = str(filename)
        self.window_closed = threading.Event()  # set if plot window closed by the user

    def closeEvent(self, *args, **kwargs):
        """ """
        self.window_closed.set()
        ImageExporter(self.ci).export(self.filename)
        super().closeEvent(*args, **kwargs)


class Plotter:
    """ """

    WINDOW_SIZE = (1200, 800)
    WINDOW_BORDER = True

    MAX_PLOTS = 4
    MAX_COLS = 2

    MAX_DATA_ITEMS: int = 20  # maximum number of traces in one plot
    SCATTER_DOT_SIZE: int = 6

    def __init__(
        self, interval: float, expt_name: str, datafile, *datasets: Dataset
    ) -> None:
        """ """
        self.interval = interval
        self.datasets = datasets
        self.header, self._header_text = None, expt_name
        self._expt_name = expt_name
        self.footer, self._footer_text = None, f"Datafile: {datafile}"
        self.filename = datafile.parent / f"{datafile.stem}.png"

        if len(datasets) > Plotter.MAX_PLOTS:
            message = f"Exceeded max number of supported plots: {Plotter.MAX_PLOTS}."
            logger.error(message)
            raise PlotterInitializationError(message)

        # Qt objects to be controlled by the Plotter
        self.app, self.layout, self.timer = None, None, None

        # Events to coordinate plotting window behaviour in plot()
        self.new_data_event = threading.Event()  # set if new data is found for plotting
        self.done_event = threading.Event()  # set if plotting is complete
        self.exit_event = threading.Event()  # to close plots by the Experiment

        self.stop_expt = False  # to stop experiment if user closes plotting window

        # run() will initialize plots in the plotting window after updating the plotspec
        self.plotspec: dict[Dataset, PlotSpec] = {}

        # open the plotting window in a separate thread
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self) -> None:
        """ """
        pg.setConfigOptions(
            antialias=True, imageAxisOrder="row-major", background="w", foreground="k"
        )

        self.app = pg.mkQApp()
        self.layout = PlotWidget(filename=self.filename, show=True)
        self.layout.showMaximized()
        self.layout.setWindowTitle("Qcore plotter")

        self.plotspec: dict[Dataset, PlotSpec] = {d: PlotSpec(d) for d in self.datasets}

        cmax = Plotter.MAX_COLS
        ht, ft = self._header_text, self._footer_text
        self.header = self.layout.addLabel(ht, colspan=cmax, size="16pt", bold=True)

        # create the plot layout based on the total number of datasets to be plotted
        if len(self.datasets) == 1:  # to ensure proper alignment of borders
            plotspec = list(self.plotspec.values())[0]
            self.layout.addItem(plotspec.plot_item, row=1, col=0, colspan=cmax)
            r = 2  # row
            if self.datasets[0].fitfn is not None:
                fit_lbl = self.layout.addLabel(row=r, col=0, colspan=cmax, size="10pt")
                plotspec.fit_label = fit_lbl
                r += 1
            ftr = self.layout.addLabel(ft, r, 0, colspan=cmax, size="10pt", bold=True)
        else:
            r, c, has_flbl = 1, 0, False  # row, column, row has fit label
            for dataset, plotspec in self.plotspec.items():
                self.layout.addItem(plotspec.plot_item, row=r, col=c)
                if dataset.fitfn is not None:
                    fit_lbl = self.layout.addLabel("", r + 1, c, size="10pt")
                    plotspec.fit_label = fit_lbl
                    has_flbl = True
                c += 1
                if c >= Plotter.MAX_COLS:
                    c = 0
                    r = r + 2 if has_flbl else r + 1
                    has_flbl = False
            if c < Plotter.MAX_COLS:
                r = r + 2 if has_flbl else r + 1
            ftr = self.layout.addLabel(ft, r, 0, colspan=cmax, size="10pt", bold=True)
        self.footer = ftr

        # setup crosshair
        for spec in self.plotspec.values():
            spec.plot_item.scene().sigMouseMoved.connect(self.mouse_moved)

        self.layout.ci.layout.setSpacing(20)
        self.layout.ci.setContentsMargins(20, 20, 20, 20)

        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.interval * 1000)

        self.app.exec()

    def plot(self, message, stop=False, exit=False) -> None:
        """ """
        if self.layout is not None and self.layout.window_closed.is_set():
            self.stop_expt = True

        self._header_text = f"{self._expt_name}{message}"
        self.new_data_event.set()
        if exit:
            time.sleep(self.interval)
            self.exit_event.set()
        if stop:
            time.sleep(self.interval)
            self.done_event.set()

    def update(self):
        """ """
        if not self.done_event.is_set():
            if self.new_data_event.is_set():
                self.new_data_event.clear()
                self.header.setText(f"{self._header_text}")

                for dataset, spec in self.plotspec.items():
                    if spec.num_data_items == 1:
                        self._plot_single(dataset, spec)
                    else:
                        self._plot_multiple(dataset, spec)

        else:
            if self.exit_event.is_set():
                self.layout.close()
            self.timer.stop()

    def mouse_moved(self, position):
        """ """
        for spec in self.plotspec.values():
            plot_item = spec.plot_item
            if plot_item.sceneBoundingRect().contains(position):
                mouse_point = plot_item.vb.mapSceneToView(position)
                x, y = mouse_point.x(), mouse_point.y()
                spacing, coord = "&nbsp;" * 128, f"[{x = :.8g}, {y = :.8g}]"
                text = f"{self._footer_text} <span>{spacing} {coord}</span>"
                self.footer.setText(text)
                cx, cy = spec.crosshair
                cx.setPos(x)
                cy.setPos(y)

    def _plot_single(self, dataset: Dataset, plotspec: PlotSpec):
        """ """
        plot_data_item = plotspec.plot_data_items[0]
        sweep_data = list(dataset.sweep_data.values())
        x = sweep_data[-1]
        y = dataset.avg
        if plotspec.plot_type == "image":
            y = sweep_data[-2]
            z = dataset.avg
            self._plot_2D(plot_data_item, x, y, z)
            plotspec.cbar.setLevels(low=np.min(z), high=np.max(z))
        elif plotspec.plot_type in ("scatter", "line"):
            self._plot_1D(plot_data_item, x, y)
            if plotspec.plot_err:
                plot_err_item = plotspec.plot_err_items[0]
                self._plot_errorbar(plot_err_item, x, y, dataset.sem)

            if dataset.fitfn is not None:
                plot_fit_item = plotspec.plot_fit_items[0]
                best_fit, fit_params = dataset.fitfn(y, x)
                self._plot_1D(plot_fit_item, x, best_fit)
                fit_str = f", ".join(f"{k}: {v:.6g}" for k, v in fit_params.items())
                plotspec.fit_label.setText(fit_str)
                dataset.best_fit, dataset.fit_params = best_fit, fit_params

    def _plot_multiple(self, dataset: Dataset, plotspec: PlotSpec):
        """ """
        sweep_data = list(dataset.sweep_data.values())
        data = dataset.avg
        x, y, err = sweep_data[-1], sweep_data[-2], dataset.sem
        all_best_fits, all_fit_params = [], {}
        for i in range(plotspec.num_data_items):
            z = data[i]
            plot_data_item = plotspec.plot_data_items[i]
            self._plot_1D(plot_data_item, x, z)
            if plotspec.plot_err:
                plot_err_item = plotspec.plot_err_items[i]
                self._plot_errorbar(plot_err_item, x, z, err[i])
            if dataset.fitfn is not None:
                plot_fit_item = plotspec.plot_fit_items[i]
                best_fit, fit_params = dataset.fitfn(z, x)
                to_round = (float, np.floating)
                ytxt = f"{y[i]:.5f}" if isinstance(y[i], to_round) else f"{y[i]}"
                all_fit_params[ytxt] = fit_params
                all_best_fits.append(best_fit)
                self._plot_1D(plot_fit_item, x, best_fit)

        dataset.best_fit = np.array(all_best_fits)
        dataset.fit_params = list(all_fit_params.values())

        fit_str = ""
        for label, fit_params in all_fit_params.items():
            fit_str += f"[{label}] "
            fit_str += f", ".join(f"{k}: {v:.6g}" for k, v in fit_params.items())
            fit_str += "<br>"
        if fit_str:
            plotspec.fit_label.setText(fit_str[:-4])

    def _plot_1D(self, plot, x, y):
        """ """
        # EDIT 19/07/25 by JC: Temp fix to ADC bug <----------------------------------------------------------------------------------------------------
        # plot.setData(x=x, y=np.real(y))

        plot.setData(x=np.real(x), y=np.real(y))


    def _plot_2D(self, plot, x, y, z):
        """ """
        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])
        width = np.abs(x[-1] - x[0]) + dx
        height = np.abs(y[-1] - y[0]) + dy
        plot.setImage(image=z, rect=(x[0] - dx / 2, y[0] - dy / 2, width, height))

    def _plot_errorbar(self, plot, x, y, err):
        """ """
        plot.setData(x=x, y=y, height=err)
