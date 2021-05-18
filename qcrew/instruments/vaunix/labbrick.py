""" """

from typing import ClassVar, NoReturn

import qcrew.instruments.vaunix.labbrick_api as vnx
from qcrew.helpers import logger
from qcrew.instruments import PhysicalInstrument


class LabBrick(PhysicalInstrument):
    """ """

    # class variable defining the parameter set for LabBrick objects
    _parameters: ClassVar[frozenset[str]] = frozenset(["frequency", "power"])

    # pylint: disable=redefined-builtin, intentional shadowing of `id`

    def __init__(self, id: int, frequency: float = None, power: float = None) -> None:
        """ """
        super().__init__(id)
        self._handle: int = self._connect()
        self._initialize(frequency, power)

    # pylint: enable=redefined-builtin

    def _connect(self) -> int:
        """ """
        try:
            device_handle = vnx.connect_to_device(self.id)
        except ConnectionError:
            logger.exception(f"Failed to connect to LB{self.id}")
            raise
        else:
            logger.info(f"Connected to LB{self.id}")
            return device_handle

    def _initialize(self, frequency: float, power: float) -> NoReturn:
        """ """
        vnx.set_use_internal_ref(self._handle, False)  # use external 10MHz reference
        self.toggle_rf()  # turn on RF, guaranteed to be off

        # if user specifies initial frequency and power, set them
        # else, get current frequency and power from device and set those
        self.frequency = frequency if frequency is not None else self.frequency
        self.power = power if power is not None else self.power

    def toggle_rf(self) -> NoReturn:
        """ """
        toggle = not vnx.get_rf_on(self._handle)
        vnx.set_rf_on(self._handle, toggle)
        logger.success(f"LB{self.id} RF is {'ON' if toggle else 'OFF'}")

    @property  # frequency getter
    def frequency(self) -> float:
        """ """
        try:
            frequency = vnx.get_frequency(self._handle)
        except ValueError:
            logger.exception(f"LB{self.id} failed to get frequency")
            raise
        else:
            logger.success(f"LB{self.id} current {frequency:.7E = } Hz")
            return frequency

    @frequency.setter
    def frequency(self, new_frequency: float) -> NoReturn:
        """ """
        try:
            frequency = vnx.set_frequency(self._handle, new_frequency)
        except (TypeError, ValueError, ConnectionError):
            logger.exception("LB{} failed to set frequency", self.id)
            raise
        else:
            logger.success(f"LB{self.id} set {frequency:.7E = } Hz")

    @property  # power getter
    def power(self) -> float:
        """ """
        try:
            power = vnx.get_power(self._handle)
        except ValueError:
            logger.exception(f"LB{self.id} failed to get power")
            raise
        else:
            logger.success(f"LB{self.id} current {power = } dBm")
            return power

    @power.setter
    def power(self, new_power: float) -> NoReturn:
        """ """
        try:
            power = vnx.set_power(self._handle, new_power)
        except (TypeError, ValueError, ConnectionError):
            logger.exception(f"LB{self.id} failed to set power")
            raise
        else:
            logger.success(f"LB{self.id} set {power = } dBm")

    def disconnect(self):
        """ """
        if vnx.get_rf_on(self._handle):
            self.toggle_rf()  # turn off RF if on

        try:
            vnx.close_device(self._handle)
        except ConnectionError:
            logger.exception(f"Failed to close LB{self.id}")
            raise
        else:
            logger.info(f"Disconnected LB{self.id}")
