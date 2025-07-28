""" """

from tkinter.messagebox import NO
from typing import Any, Union

import numpy as np

from qcore.helpers.logger import logger
from qcore.libs.data_fns import DATAFN_MAP
from qcore.libs.fit_fns import FITFN_MAP
from qcore.libs.qua_macros import QuaVariable
from qcore.variables.sweeps import Sweep


class DatasetInitializationError(Exception):
    """ """


class Dataset(QuaVariable):
    """Class that allows users to specify Datasets for handling data obtained from Experiments.

    We save raw Dataset data and plot averaged data. Assume the outermost axis to be the averaging axis.

    list of acceptable kwargs, their meanings, and default values:
    - data (initial data assigned to the dataset default: np.zeros(dataset.shape))
    - dtype (data type the values are saved to the datafile with, default: float)
    - units (units attribute the dataset is saved to the datafile with, default: None)
    - fitfn
    - datafn
    - inputs
    - datafn_args
    - plot_args
        - plot_type: ("scatter", "line", "image), default = "scatter"
        - plot_err: whether or not to show errorbars, default = True
        - xlabel: str
        - ylabel: str
        - title: str
        - cmap (for image type plots only), default="viridis"
    - buffer_shape (for qua stream processing)
    """

    def __init__(
        self,
        name: str,  # name of the dataset, as it will appear in the datafile
        axes: list[Union[Sweep, int]] = None,  # the dataset's dimensions
        stream: bool = False,  # False to not stream data from OPX, True to stream non-adc data, and 1 or 2 to stream adc data from input ports 1 or 2 respectively
        save: bool = False,  # whether or not to save this Dataset to a datafile
        plot: bool = False,  # whether or not to plot this Dataset during live plot
        **kwargs,  # control data processing, fitting, plotting, saving
    ) -> None:
        """ """
        self.name = name

        self._stream = stream
        self.save = save
        self.plot = plot

        self._axes = None
        self.axes = axes
        self.index = None  # for saving data
        self.dtype = kwargs.get("dtype", float)
        self.units = kwargs.get("units", "A.U.")

        self._datafn = None
        datafn = kwargs.get("datafn")
        if datafn is not None:
            self.datafn = datafn
        self.inputs = kwargs.get("inputs", ())
        self.datafn_args = kwargs.get("datafn_args", {})
        self.data = kwargs.get("data")
        self.avg, self.sem, self.var, self.std, self.count = None, None, None, None, 0

        self._fitfn = None
        fitfn = kwargs.get("fitfn")
        if fitfn is not None:
            self.fitfn = fitfn
        self.best_fit, self.fit_params = None, None

        self.plot_args = kwargs.get("plot_args", {})

        buffer = kwargs.get("buffer")
        super().__init__(self.dtype, stream=stream, tag=name, buffer=buffer)

    def __repr__(self) -> str:
        """ """
        return f"{self.__class__.__name__} '{self.name}'"

    @property
    def axes(self):
        """ """
        return self._axes

    @axes.setter
    def axes(self, value):
        """ """
        self._axes = value

    @property
    def datafn(self):
        """ """
        return self._datafn

    @datafn.setter
    def datafn(self, value: str):
        """ """
        try:
            self._datafn = DATAFN_MAP[value]
        except KeyError:
            valid_datafns = list(DATAFN_MAP.keys())
            message = f"Datafn '{value}' does not exist. {valid_datafns = }."
            logger.error(message)
            raise DatasetInitializationError(message)

    @property
    def fitfn(self):
        """ """
        return self._fitfn

    @fitfn.setter
    def fitfn(self, value: str):
        """ """
        try:
            self._fitfn = FITFN_MAP[value]
        except KeyError:
            valid_fitfns = list(FITFN_MAP.keys())
            message = f"Fit function '{value}' does not exist. {valid_fitfns = }."
            logger.error(message)
            raise DatasetInitializationError(message)

    @property
    def shape(self):
        """ """
        if self._axes is None:
            return
        return tuple(i.length if isinstance(i, Sweep) else i for i in self._axes)

    @property
    def sweep_data(self):
        """ """
        if self._axes is None:
            return
        sdata = {}
        for idx, ax in enumerate(self.axes):
            if isinstance(ax, Sweep):
                sdata[ax.name] = ax.data
            else:
                sdata[str(idx)] = np.arange(1, ax + 1, 1, dtype=int)
        return sdata

    @property
    def metadata(self) -> dict[str, Any]:
        """ """
        return {"name": self.name, "dtype": self.dtype, "units": self.units}

    def initialize(self, axes: list[Sweep]) -> None:
        """ """
        self.axes = axes
        shape = list(self.shape)
        self.data = np.zeros(shape)
        self.avg = np.average(self.data, axis=0)
        self.std = np.average(self.data, axis=0)
        self.sem = np.average(self.data, axis=0)
        self.var = np.average(self.data, axis=0)
        if self.stream:
            shape.pop(0)
            self.buffer = shape

    def update(self, datasets, pnum, inum) -> None:
        """ """
        # update only if new data found
        if pnum == inum:
            return

        if self.datafn is None:  # primary dataset
            self.data, avg = datasets
        else:  # derived dataset
            input_data = [d.data for d in datasets]
            self.data = self.datafn(input_data, **self.datafn_args)
            input_avg = [d.avg if isinstance(d, Dataset) else d.data for d in datasets]
            avg = self.datafn(input_avg, **self.datafn_args)

        # update index of next batch of data to be inserted in the datafile
        self.index = (slice(pnum, inum), ...)

        # calculate avg and stderr
        k = pnum + inum
        estimator = (self.data - self.avg) * (self.data - avg)
        self.var = self.var * (pnum - 1)
        self.var = (self.var + np.sum(estimator, axis=0)) / (inum - 1)
        self.avg, self.std, self.sem = avg, np.sqrt(self.var), np.sqrt(self.var / k)
