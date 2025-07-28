""" """

from __future__ import annotations

from ctypes import CDLL, c_int
from pathlib import Path

from qcore.instruments.instrument import Instrument, ConnectionError
from qcore.variables.parameter import Parameter

# DLL driver must be placed in the same folder as this file
DLL = CDLL(str(Path(__file__).parent / "vaunix_lms.dll"))

UNIT_FREQUENCY = 10.0  # LMS encodes frequency as an integer of 10Hz steps
UNIT_POWER = 0.25  # LMS encodes power level as an integer of 0.25dB steps


def check_frequency(value: float, lms: LMS) -> bool:
    """ """
    min, max = lms.min_frequency, lms.max_frequency
    LMS.frequency.hint = f"[{min:.2E}, {max:.2E}]"
    return min <= value <= max


def to_frequency(value: int) -> float:
    """Convert LMS coded frequency to actual frequency"""
    return value * UNIT_FREQUENCY


def from_frequency(value: float) -> int:
    """Convert frequency to LMS coded frequency"""
    return int(float(value) / UNIT_FREQUENCY)


def check_power(value: float, lms: LMS) -> bool:
    """ """
    min, max = lms.min_power, lms.max_power
    LMS.power.hint = f"[{min}, {max}]"
    return min <= value <= max


def to_power(value: int) -> float:
    """Convert LMS encoded power to actual power"""
    return value * UNIT_POWER


def from_power(value: float) -> int:
    """Convert actual power to LMS encoded power"""
    return int(float(value) / UNIT_POWER)


class LMS(Instrument):
    """ """

    clocked: bool = Parameter()
    output: bool = Parameter()
    frequency: float = Parameter(bounds=check_frequency)
    power: float = Parameter(bounds=check_power)

    def __init__(
        self,
        name: str,
        id: str,
        frequency: float = 6e9,
        power: float = 0.0,
        output: bool = False,
        **parameters,
    ) -> None:
        """ """
        self._handle = None
        super().__init__(
            id, name=name, frequency=frequency, power=power, output=output, **parameters
        )

        DLL.fnLMS_SetTestMode(False)  # we are using actual hardware
        DLL.fnLMS_SetUseInternalRef(self._handle, False)  # use external 10MHz reference
        print(f"{self} is ready to use!")

    def _errorcheck(self, errorcode: int) -> None:
        """Only if we get bad values during setting params"""
        if errorcode:  # non-zero return values indicate error
            message = f"Got {errorcode = } from {self}, reconnect device."
            self._handle = None
            raise ConnectionError(message)

    @property
    def status(self) -> bool:
        """A connected LMS status code has 'b' as its final hex digit"""
        code = hex(DLL.fnLMS_GetDeviceStatus(self._handle))
        status = code[-1] == "b"
        if not status:
            self._handle = None
        return status

    def connect(self) -> None:
        """ """
        # close any existing connection
        if self._handle is not None:
            self.disconnect()

        numdevices = DLL.fnLMS_GetNumDevices()
        deviceinfo = (c_int * numdevices)()
        DLL.fnLMS_GetDevInfo(deviceinfo)
        ids = [DLL.fnLMS_GetSerialNumber(deviceinfo[i]) for i in range(numdevices)]
        id = int(self.id)
        if id in ids:  # LMS is found, try opening it
            handle = deviceinfo[ids.index(id)]
            error = DLL.fnLMS_InitDevice(handle)
            if not error:  # 0 indicates successful device initialization
                self._handle = handle
                print(f"Connected to {self}!")
                return
            raise ConnectionError(f"Failed to connect {self}.")
        raise ConnectionError(f"{self} is not available for connection.")

    def disconnect(self):
        """ """
        self._errorcheck(DLL.fnLMS_CloseDevice(self._handle))
        self._handle = None
        print(f"Disconnected {self}!")

    @clocked.getter
    def clocked(self) -> bool:
        """The hex code for PLL_LOCKED flag is 0x00000040 in vnx_LMS_api.h"""
        value = bool(int(hex(DLL.fnLMS_GetDeviceStatus(self._handle))[-2]))
        return value

    @output.getter
    def output(self) -> bool:
        """ """
        value = bool(DLL.fnLMS_GetRF_On(self._handle))
        return value

    @output.setter
    def output(self, value: bool) -> None:
        """ """
        self._errorcheck(DLL.fnLMS_SetRFOn(self._handle, value))
        print(f"Set {self} output {value = }.")

    @frequency.getter
    def frequency(self) -> float:
        """ """
        value = to_frequency(DLL.fnLMS_GetFrequency(self._handle))
        return value

    @frequency.setter
    def frequency(self, value: float) -> None:
        """ """
        self._errorcheck(DLL.fnLMS_SetFrequency(self._handle, from_frequency(value)))
        print(f"Set {self} frequency {value = }.")

    @power.getter
    def power(self) -> float:
        """ """
        value = to_power(DLL.fnLMS_GetAbsPowerLevel(self._handle))
        return value

    @power.setter
    def power(self, value: float) -> None:
        """ """
        self._errorcheck(DLL.fnLMS_SetPowerLevel(self._handle, from_power(value)))
        print(f"Set {self} power {value = }.")

    @property
    def min_frequency(self) -> float:
        """ """
        return to_frequency(DLL.fnLMS_GetMinFreq(self._handle))

    @property
    def max_frequency(self) -> float:
        """ """
        return to_frequency(DLL.fnLMS_GetMaxFreq(self._handle))

    @property
    def min_power(self) -> float:
        """ """
        return to_power(DLL.fnLMS_GetMinPwr(self._handle))

    @property
    def max_power(self) -> float:
        """ """
        return to_power(DLL.fnLMS_GetMaxPwr(self._handle))
