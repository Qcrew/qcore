""" """

from __future__ import annotations

from math import ceil
from typing import Any, Union, Type

import numpy as np

from qm import qua

from qcore.helpers.logger import logger
from qcore.libs.qua_macros import QuaVariable


class Sweep:
    """Sweep specification class, meant for user to conveniently specify sweeps, the specification will be used by Experiment to instantiate BaseSweep classes. We call this class 'Sweep' for user convenience, it is more of a 'SweepInfo' class."""

    def __init__(
        self,
        name: str,  # name of sweep variable
        target: str = None,  # name of an object in config whose param is to be swept
        dtype: Type[Union[float, int, str]] = float,  # sweep data type
        units: str = None,  # units attribute sweep points dataset is saved with
        save: bool = True,  # whether or not to save this Sweep to a datafile
        points: list[Union[float, int, str]] = None,  # a list of discrete sweep points
        start: Union[float, int] = 1,  # start point for arange- or linspace-like sweeps
        stop: Union[float, int] = None,  # end point for arange- or linspace-like sweeps
        step: Union[float, int] = 1,  # point spacing for arange-like sweeps
        num: int = None,  # number of sweep points for np.linspace-like sweeps
        endpoint: bool = True,  # whether or not to include end point in sweep
        kind: str = "lin",  # choose linear ("lin") or logarithmic ("log") sweep spacing
    ) -> None:
        """ """
        self.name = name
        self._sweep: BaseSweep = None  # BaseSweep based on this Sweep specification
        self.start, self.stop, self.step, self.num = start, stop, step, num
        self.endpoint = endpoint
        self.points = points
        self.kind = kind
        self.target = target
        self.dtype = dtype
        self.units = units
        self.save = save

    @property
    def sweep(self) -> BaseSweep:
        """ """
        # resolution order for obtaining BaseSweep: PointsSweep > EvenSweep > RangeSweep

        pts_kws = {"start": self.start, "stop": self.stop, "endpoint": self.endpoint}

        # PointsSweep has discrete points - all int, float, str
        if isinstance(self.points, (list, tuple, set)):
            if all(isinstance(x, (int, float, str)) for x in self.points):
                self.dtype = np.dtype(type(self.points[0]))
                sweep_pts = DiscretePoints(points=self.points)
        elif isinstance(self.num, int):  # EvenlySpacedSweepPoints: LinSweep or LogSweep
            if self.kind == "lin":
                sweep_pts = LinSpacedPoints(num=self.num, dtype=self.dtype, **pts_kws)
            elif self.kind == "log":
                sweep_pts = LogSpacedPoints(num=self.num, dtype=self.dtype, **pts_kws)
        elif isinstance(self.stop, (int, float)):
            sweep_pts = RangePoints(step=self.step, dtype=self.dtype, **pts_kws)
        else:
            raise ValueError(f"Badly specified sweep '{self.name}'.")

        if self.target is None:
            self._sweep = QuaSweep(self.name, self.dtype, sweep_pts, units=self.units, save=self.save)
        else:
            self._sweep = QcoreSweep(self.target, units=self.units, save=self.save, name=self.name, dtype=self.dtype, sweep_points=sweep_pts)

        return self._sweep

class SweepPoints:
    """ """

    @property
    def data(self) -> np.ndarray:
        """ """
        raise NotImplementedError("Subclass(es) to define data")

    @property
    def metadata(self) -> dict[str, Any]:
        """ """
        raise NotImplementedError("Subclass(es) to define metadata")


class BaseSweep:
    """Base class for all concrete Sweep types to inherit from and implement"""

    def __init__(
        self,
        name: str,
        dtype,
        sweep_points: SweepPoints,
        units: str,
        save: bool,
    ) -> None:
        """ """
        self.name = name
        self.dtype = dtype
        self.sweep_points = sweep_points
        self.units = units
        self.save = save
        logger.info(f"Initialized {self} with {self.metadata}.")

    def __repr__(self) -> str:
        """ """
        return (
            f"{self.__class__.__name__} '{self.name}' "
            f"[{self.sweep_points.__class__.__name__}]"
        )

    @property
    def data(self) -> np.ndarray:
        """ """
        return self.sweep_points.data

    @property
    def metadata(self) -> dict[str, Any]:
        """ """
        mdata = {
            "name": self.name,
            "dtype": self.dtype,
            "units": self.units,
        }
        return {**mdata, **self.sweep_points.metadata}

    @property
    def length(self) -> int:
        """ """
        return len(self.data)

    @property
    def shape(self) -> tuple[int]:
        """ """
        return (self.length,)


class QuaSweep(BaseSweep, QuaVariable):
    """Only QuaSweeps are QUA-compatible."""

    def __init__(self, name, dtype, sweep_pts, units=None, save=False) -> None:
        """ """
        self._data = None  # not None during live saving
        BaseSweep.__init__(self, name, dtype, sweep_pts, units, save)
        QuaVariable.__init__(self, dtype, stream=True, tag=name, buffer=self.shape)
        self.index = ...  # for live saving

    def generate_loop(self):
        """ """
        var, dtype = self.qua_variable, self.dtype
        if isinstance(self.sweep_points, DiscretePoints):
            return qua.for_each_, var, self.data.tolist()
        elif isinstance(self.sweep_points, RangePoints):
            start, stop = self.sweep_points.start, self.sweep_points.stop
            return qua.for_, var, start, var < stop, var + self.sweep_points.step
        elif isinstance(self.sweep_points, LinSpacedPoints):
            data, endpoint = self.sweep_points.data, self.sweep_points.endpoint
            if len(data) > 1:
                start, stop, step = data[0], data[-1], data[1] - data[0]
                pts = RangePoints(start, stop, step, endpoint, dtype)
                return qua.for_, var, pts.start, var < pts.stop, var + pts.step
            else:
                return qua.for_each_, var, data.tolist()
        else:
            message = f"Unsupported Sweep -> QUA loop conversion for {self}."
            logger.error(message)
            raise TypeError(message)

    @property
    def data(self):
        """ """
        return super().data if self._data is None else self._data

    def update(self, data) -> None:
        """ """
        self._data = data


class QcoreSweep(BaseSweep):
    """ """

    def __init__(self, target, **kwargs) -> None:
        """ """
        self.target = target
        super().__init__(**kwargs)


class DiscretePoints(SweepPoints):
    """sweep with discrete points specified by the user"""

    def __init__(self, points) -> None:
        """ """
        self._points = points

    @property
    def data(self):
        """ """
        return np.array(self._points)

    @property
    def metadata(self):
        """ """
        return {}


class RangePoints(SweepPoints):
    """mimic numpy.arange with option to include/exclude endpoint"""

    def __init__(self, start, stop, step, endpoint, dtype) -> None:
        """ """
        self.dtype = dtype
        self.start = dtype(start)
        self.endpoint = endpoint
        self.step = dtype(step)
        self.stop = stop + step / 2 if endpoint else stop - step / 2
        if dtype is int:
            self.stop = ceil(stop + step / 2) if endpoint else ceil(stop - step / 2)

    @property
    def data(self):
        """ """
        if self.endpoint:
            return np.arange(self.start, self.stop, self.step, dtype=self.dtype)
        else:
            return np.arange(self.start, self.stop, self.step, dtype=self.dtype)

    @property
    def metadata(self):
        """ """
        return {
            "start": self.start,
            "stop": self.stop,
            "step": self.step,
            "endpoint": self.endpoint,
        }


class EvenlySpacedPoints(SweepPoints):
    """evenly spaced sweep"""

    def __init__(self, start, stop, num, endpoint, fn, dtype) -> None:
        """ """
        self.start = start
        self.stop = stop if stop is not None else num
        self.num = num
        self.endpoint = endpoint
        self.fn = fn  # np.linspace or np.logspace
        self.dtype = dtype

    @property
    def data(self):
        """ """
        return self.fn(
            start=self.start,
            stop=self.stop,
            num=self.num,
            endpoint=self.endpoint,
            dtype=self.dtype,
        )

    @property
    def metadata(self):
        """ """
        kind = None
        if self.fn is np.linspace:
            kind = "lin"
        elif self.fn is np.logspace:
            kind = "log"
        return {
            "start": self.start,
            "stop": self.stop,
            "num": self.num,
            "endpoint": self.endpoint,
            "kind": kind,
        }


class LinSpacedPoints(EvenlySpacedPoints):
    """linear numpy linspace sweep"""

    def __init__(self, **kwargs) -> None:
        super().__init__(fn=np.linspace, **kwargs)


class LogSpacedPoints(EvenlySpacedPoints):
    """logarithmic numpy logspace sweep"""

    def __init__(self, **kwargs) -> None:
        super().__init__(fn=np.logspace, **kwargs)
