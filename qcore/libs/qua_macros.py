"""library of QUA macros, wrappers for QUA"""

from qm import qua
from qm.qua.lib import Cast
from qm.qua._dsl import _Variable, _Expression

from qcore.helpers.logger import logger
from qcore.modes import Readout


def align(*modes):
    """ """
    qua.align(*(mode.name for mode in modes))


def wait(duration: float, *modes):
    """duration to be supplied in nanoseconds and is converted to clock cycles"""
    logger.warning("Wait times will be rounded to nearest multiple of 4 nanoseconds.")
    if isinstance(duration, _Variable) or isinstance(duration, _Expression):
        qua.wait(Cast.to_int(duration / 4), *(mode.name for mode in modes))
    else:
        qua.wait(int(duration / 4), *(mode.name for mode in modes))


def reset_frame(*modes):
    qua.reset_frame(*(mode.name for mode in modes))


def reset_phase(*modes):
    for mode in modes:
        qua.reset_phase(mode.name)


def update_frequency(mode, value, units="Hz", keep_phase=False):
    qua.update_frequency(mode.name, value, units=units, keep_phase=keep_phase)


def initialize_qubit(
    rr: Readout,
    readout_pulse: str,
    demod_type: str,
    threshold_g: float,
    wait_time: int,
    ro_ampx: float = 1.0,
    n_consecutive: int = 3,
):

    I_temp = qua.declare(qua.fixed)
    Q_temp = qua.declare(qua.fixed)
    counter = qua.declare(int)
    qua.assign(counter, 0)
    with qua.while_(counter < n_consecutive):
        rr.measure(readout_pulse, (I_temp, Q_temp), ampx=ro_ampx, demod_type=demod_type)
        # increase counter if qubit is in g, reset it to 0 otherwise
        with qua.if_(I_temp < threshold_g):
            qua.assign(counter, counter + 1)
        with qua.else_():
            qua.assign(counter, 0)

        # wait for RR to reset
        wait(wait_time, rr.name)


class StreamProcessingError(Exception):
    """ """


class QuaVariable:
    """ """

    def __init__(self, dtype, stream=False, value=None, tag=None, buffer=None) -> None:
        """ """
        super().__init__()
        self.dtype = dtype

        # set stream to True to stream non-adc data from the OPX, False to not stream data from the OPX at all, and 1 or 2 to stream adc data from the OPX analog input ports 1 or 2 respectively
        self.stream = stream
        self.buffer = buffer

        self.nominal_value = value
        self.tag = tag
        self.qua_variable = None
        self.qua_stream = None

    @property
    def is_adc_trace(self) -> bool:
        """ """
        return False if isinstance(self.stream, bool) else self.stream

    def declare_variable(self):
        """call in qua variable declaration section of qua program scope"""
        if self.stream:
            self.qua_variable = qua.declare(self.dtype, value=self.nominal_value)
            return self.qua_variable

    def declare_stream(self):
        """call in qua variable declaration section of qua program scope"""
        if self.stream:
            self.qua_stream = qua.declare_stream(adc_trace=self.is_adc_trace)
            return self.qua_stream

    def save_to_stream(self):
        """this goes at the end of the pulse sequence before start of next loop iteration"""
        if self.stream is True:
            qua.save(self.qua_variable, self.qua_stream)

    def process_stream(self) -> None:
        """ """
        if not self.stream:
            return

        adc_trace = self.is_adc_trace
        if adc_trace == 1:
            self.qua_stream.input1().save_all(self.tag)
            self.qua_stream.input1().average().save(f"{self.tag}_avg")
        elif adc_trace == 2:
            self.qua_stream.input2().save_all(self.tag)
            self.qua_stream.input2().average().save(f"{self.tag}_avg")
        elif not adc_trace and hasattr(self, "sweep_points"):  # is sweep
            self.qua_stream.buffer(*self.buffer).save(self.tag)
        elif not adc_trace:  # is dataset
            self.qua_stream.buffer(*self.buffer).save_all(self.tag)
            self.qua_stream.buffer(*self.buffer).average().save(f"{self.tag}_avg")
        else:
            message = f"Failed to process stream for qua variable '{self.tag}'."
            logger.error(message)
            raise StreamProcessingError(message)
