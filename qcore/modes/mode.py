""" """
from typing import Any, Union, Tuple

from qm import qua
from qm.qua._dsl import _Variable

from qcore.modes.rf_switch import RFSwitch
from qcore.helpers.logger import logger
from qcore.pulses.pulse import Pulse
from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.pulses.readout_pulse import ReadoutPulse
from qcore.resource import Resource

FEMPort = Tuple[int, int]

class Mode(Resource):
    """ """

    PORTS_KEYS = ("I", "Q", "out1", "out2")
    OFFSETS_KEYS = (*PORTS_KEYS, "G", "P")
    RF_SWITCH_DIGITAL_MARKER = "RFSWITCH_ON"

    def __init__(
        self,
        name: str,
        lo_name: str,
        ports: dict[str, int],
        core: str = None, # Adrian 10/10/2025
        int_freq: float = -50e6,
        upconverter: int = None,
        **parameters,
    ) -> None:
        """ """
        self.octave_mixed: bool = False
        self.lo_name: str = str(lo_name)
        self.int_freq: float = int_freq
        self.core: str = core # Adrian 10/10/2025
        self.upconverter: int = upconverter
        self._ports: dict[str, Union[int, FEMPort]] = dict.fromkeys(Mode.PORTS_KEYS)
        self._mixer_offsets: dict[str, float] = dict.fromkeys(self.OFFSETS_KEYS, 0.0)
        self._rf_switch: RFSwitch = None
        self._rf_switch_on: bool = False
        self._operations: dict[str, Pulse] = {}
        self._pulse_op_map: dict[str, str] = {}

        super().__init__(name=name, ports=ports, **parameters)

    def snapshot(self, flatten=False) -> dict[str, Any]:
        """ """
        snapshot = super().snapshot()
        if flatten:
            flat_ops = {
                k: p.snapshot(flatten=True) for k, p in snapshot["operations"].items()
            }
            snapshot["operations"] = flat_ops
            rf_switch = snapshot["rf_switch"]
            if rf_switch is not None:
                snapshot["rf_switch"] = rf_switch.snapshot()
        return snapshot

    @property
    def ports(self) -> dict[str, int]:
        """ """
        return self._ports.copy()

    @ports.setter
    def ports(self, value: dict[str, int]) -> None:
        """ """
        try:
            for key in value.keys():
                if key not in self.PORTS_KEYS:
                    message = f"Invalid port {key = }, valid keys: {self.PORTS_KEYS}."
                    raise KeyError(message)
        except (AttributeError, TypeError):
            message = f"Expect {dict[str, int]} with keys: {self.PORTS_KEYS}."
            raise ValueError(message) from None

        if value:
            self._ports = value
            logger.debug(f"Set {self} ports: {value}.")

    def has_mixed_inputs(self) -> bool:
        """ """
        return self._ports.get("I") is not None and self._ports.get("Q") is not None

    @property
    def mixer_offsets(self) -> dict[str, float]:
        """ """
        return self._mixer_offsets.copy()

    @mixer_offsets.setter
    def mixer_offsets(self, value: dict[str, float]) -> None:
        """ """
        try:
            for key in value.keys():
                if key not in self.OFFSETS_KEYS:
                    msg = f"Invalid {key = }, valid offset keys: {self.OFFSETS_KEYS}."
                    raise KeyError(msg)
        except (AttributeError, TypeError):
            message = f"Expect {dict[str, float]} with keys: {self.OFFSETS_KEYS}."
            raise ValueError(message) from None

        if value:
            self._mixer_offsets = value
            logger.debug(f"Set {self} mixer offsets: {value}.")

    @property
    def rf_switch(self) -> RFSwitch:
        """ """
        return self._rf_switch

    @rf_switch.setter
    def rf_switch(self, value: RFSwitch) -> None:
        """ """
        if value is not None and not isinstance(value, RFSwitch):
            raise ValueError(f"Invalid {value = }, must be of {RFSwitch}")
        self._rf_switch = value
        logger.debug(f"Set {self} rf switch: {value}.")

    @property
    def rf_switch_on(self) -> bool:
        """ """
        return self._rf_switch_on

    @rf_switch_on.setter
    def rf_switch_on(self, value: bool) -> None:
        """ """
        if self._rf_switch is not None:
            for operation in self._operations.values():
                if not isinstance(operation, ReadoutPulse):
                    marker = DigitalWaveform(self.RF_SWITCH_DIGITAL_MARKER)
                    operation.digital_marker = marker if value else None
            self._rf_switch_on = bool(value)

    @property
    def operations(self) -> dict[str, Pulse]:
        """ """
        return self._operations.copy()

    @operations.setter
    def operations(self, value: Union[list[Pulse], dict[str, Pulse]]) -> None:
        """ """
        if isinstance(value, list):
            pulse_names = [p.name for p in value]
            pulses = value
        elif isinstance(value, dict):
            pulse_names = list(value.keys())
            pulses = list(value.values())
        else:
            raise ValueError(f"Setter expects {list[Pulse]} or {dict[str, Pulse]}.")

        ops = {}
        for pulse_name, pulse in zip(pulse_names, pulses):
            if not isinstance(pulse, Pulse):
                raise ValueError(f"Invalid value '{pulse}', must be of {Pulse}")
            ops[pulse_name] = pulse

        self._operations = ops
        self._pulse_op_map = {p.name: k for k, p in self._operations.items()}
        logger.debug(f"Set {self} operations '{pulse_names}'.")

    def add_operations(self, *operations: Pulse) -> None:
        """ """
        if self._rf_switch is not None:
            for operation in operations:
                if not isinstance(operations, ReadoutPulse):
                    marker = DigitalWaveform(self.RF_SWITCH_DIGITAL_MARKER)
                    operation.digital_marker = marker

        self.operations = [*self._operations.values(), *operations]

    def remove_operations(self, *names: str) -> None:
        """ """
        for key in names:
            if key in self._operations:
                operation = self._operations[key]
                del self._operations[key]
                logger.debug(f"Removed {self} '{key}' {operation = }.")
            else:
                logger.warning(f"Operation '{key}' does not exist for {self}.")

    def get_operations(self, *names: str) -> list[Pulse]:
        """ """
        return [self._operations[k] for k in names if k in self._operations]

    def play(self, pulse: Pulse, constant_length = 1, ampx=1.0, phase=0.0, **kwargs) -> None:
        """ """
        op_name = self._pulse_op_map[pulse.name]

        try:
            num_ampxs = len(ampx)
            if num_ampxs != 4:
                logger.error("Ampx must be a sequence of 4 values")
                raise ValueError(f"Invalid ampx value count, expect 4, got {num_ampxs}")
        except TypeError:
            num_ampxs = 1

        if isinstance(phase, _Variable):
            # assume user wants to do phase rotation if phase is supplied as a QUA variable
            qua.frame_rotation_2pi(phase, self.name)
        elif phase:
            qua.frame_rotation_2pi(phase, self.name)

        if num_ampxs == 1:
            qua.play(op_name * qua.amp(ampx), self.name, **kwargs)
        else:
            qua.play(op_name * qua.amp(*ampx), self.name, **kwargs)
            
        if isinstance(phase, _Variable):
            qua.frame_rotation_2pi(-phase, self.name) # Adrian: this was positive, but idk why
        elif phase:
            qua.frame_rotation_2pi(-phase, self.name)
