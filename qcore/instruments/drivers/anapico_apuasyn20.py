import time

import numpy as np
import pyvisa

from qcore.instruments.instrument import Instrument, ConnectionError


class APUASYN20(Instrument):
    def __init__(
        self,
        id: str,
        name: str,
        channel: int = 1,
        frequency: float = 0.0,
        phase: float = 0.0,
        power: float = 0.0,
        output: bool = False,
    ) -> None:
        """ """
        self._handle: pyvisa.resource.Resource = None
        super().__init__(
            id=id,
            name=name,
            channel=channel,
            frequency=frequency,
            phase=phase,
            power=power,
            output=output,
        )

    def connect(self) -> None:
        """ """
        if self._handle is not None:
            self.disconnect()
        resource_name = f"USB0::0x03EB::0xAFFF::{self.id}::INSTR"
        try:
            self._handle = pyvisa.ResourceManager().open_resource(resource_name)
        except pyvisa.errors.VisaIOError as err:
            details = f"{err.abbreviation} : {err.description}"
            raise ConnectionError(f"Failed to connect {self}, {details = }") from None

    def disconnect(self) -> None:
        """ """
        self._handle.close()

    @property
    def status(self) -> bool:
        """ """
        try:
            self._handle.query("*IDN?")
        except (pyvisa.errors.VisaIOError, pyvisa.errors.InvalidSession):
            return False
        else:
            return True

    @property
    def channel(self) -> int:
        """Returns the current active channel"""
        return int(self._handle.query(f":SEL?"))

    @channel.setter
    def channel(self, value: int) -> None:
        """Sets active channel"""
        self._handle.write(f":SEL {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    @property
    def frequency(self) -> float:
        """Returns freq in Hz"""
        return float(self._handle.query(f":FREQ:CW?"))

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Writes frequency in Hz"""
        self._handle.write(f":FREQ:CW {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    @property
    def phase(self) -> float:
        """Returns phase in rad"""
        return float(self._handle.query(f":PHAS:ADJ?"))

    @phase.setter
    def phase(self, value: float):
        """Writes phase in rad"""
        self._handle.write(f":PHAS:ADJ {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    @property
    def power(self) -> float:
        """Returns power in dBm"""
        return float(self._handle.query(f":POW?"))

    @power.setter
    def power(self, value: float):
        """Sets power in dBm"""
        self._handle.write(f":POW {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    @property
    def output(self) -> bool:
        """ """
        return bool(self._handle.query(f"OUTP?"))

    @output.setter
    def output(self, value: bool):
        """ """
        if value:
            self._handle.write(f"OUTP ON")
        else:
            self._handle.write(f"OUTP OFF")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    def get_channel_freq(self, channel: int) -> float:
        """Returns freq of channel in Hz"""
        return float(self._handle.query(f":SOUR{channel}:FREQ:CW?"))

    def set_channel_freq(self, channel: int, value: float) -> float:
        """Sets freq of channel in Hz"""
        self._handle.write(f":SOUR{channel}:FREQ:CW {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    def get_channel_pow(self, channel: int) -> float:
        """Returns power of channel in dBm"""
        return float(self._handle.query(f":SOUR{channel}:POW?"))

    def set_channel_pow(self, channel: int, value: float) -> float:
        """Sets power of channel in dBm"""
        self._handle.write(f":SOUR{channel}:POW {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    def get_channel_phase(self, channel: int) -> float:
        """Returns phase of channel in rad"""
        return float(self._handle.query(f":SOUR{channel}:PHAS?"))

    def set_channel_phase(self, channel: int, value: float) -> float:
        """Sets phase of channel in rad"""
        self._handle.write(f":SOUR{channel}:PHAS {value}")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")

    def setup_pulse_mod(self, channel: int) -> bool:
        """ """

        # Sets reference osc to an external source (Rubidium clock)
        self._handle.write(f":SOUR{channel}:ROSC:SOUR EXT")
        # Make sure that freq change happens immediately and not on trigger
        self._handle.write(f":SOUR{channel}:FREQ:TRIG OFF")
        # Set all subsystems to fixed
        self._handle.write(f":SOUR{channel}:FREQ:MODE CW")
        self._handle.write(f":SOUR{channel}:POW:MODE CW")
        self._handle.write(f":SOUR{channel}:PHASE:MODE CW")

        # Turn off other modulation methods
        self._handle.write(f":SOUR{channel}:AM:STAT OFF")
        self._handle.write(f":SOUR{channel}:FM:STAT OFF")
        self._handle.write(f":SOUR{channel}:PM:STAT OFF")

        # Turn on pulse modulation and enable output
        self._handle.write(f":SOUR{channel}:PULM:SOUR EXT")
        self._handle.write(f":SOUR{channel}:PULM:STAT ON")
        self._handle.write(f":OUTP {channel} ON")
        self._handle.write(f":OUTP:BLAN {channel} OFF")
        # Synchronize (wait until all previous commands have been executed completely)
        self._handle.query("*OPC?")
        pulse_mode_src = str(self._handle.query(f":SOUR{channel}:PULM:SOUR?"))
        pulse_mode_on = bool(self._handle.query(f":SOUR{channel}:PULM:STAT?"))

        return (pulse_mode_src == "EXT") and pulse_mode_on

    def setup_freq_sweep(self, channel: int, start: float, stop: float, step: float):
        """ """
        self._handle.write(f"SOUR{channel}:FREQ:MODE SWE")
        self._handle.write(f"SOUR{channel}:FREQ:STAR {start}")
        self._handle.write(f"SOUR{channel}:FREQ:STOP {stop}")
        self._handle.write(f"SOUR{channel}:FREQ:STEP {step}")
        self._handle.write(f":TRIG:TYPE POINT")
