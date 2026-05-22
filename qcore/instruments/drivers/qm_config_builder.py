""" """
import json
from typing import Union, Optional

from qcore.instruments.drivers.qm_config import QMConfig
from qcore.instruments.drivers.qm_config_opx1000 import QMConfigOPX1000
from qcore.instruments.drivers.qm_config_opx_plus import QMConfigOPXPlus
from qcore.instruments.drivers.qm_octave_dummy import Octave
from qcore.instruments.drivers.qm_opx import OPX
from qcore.modes.mode import Mode
from qcore.modes.readout import Readout
from qcore.instruments.drivers.vaunix_lms import LMS

LO = Union[LMS, Octave]


class QMConfigBuildingError(Exception):
    """ """


class QMConfigBuilder:
    """ """

    def __init__(self) -> None:
        """ """
        self._config: QMConfig = None  # built by build_config()
        self._modes: tuple[Mode] = None
        self._lo_frequencies: dict[str, float] = {}
        self._octaves: dict[str, Octave] = {}

    def build_config(
        self, modes: tuple[Mode], los: tuple[LO], opx: OPX = None, controllers_info: Optional[dict] = None
    ) -> QMConfig:
        """ """
        try:
            self._check_modes(*modes)
            self._check_local_oscillators(*los)
            self._opx = opx
        except TypeError:
            message = f"Expect tuple arguments for 'modes' and 'los'."
            raise QMConfigBuildingError(message) from None
        else:
            if opx is not None:
                if self.uses_opx_plus():
                    self._config = QMConfigOPXPlus()
                elif self.uses_opx1000():
                    self._config = QMConfigOPX1000(controllers_info)
                else:
                    raise NotImplementedError()
            self._build_config()
            return self._config

    def _build_config(self) -> None:
        """ """
        config, modes, lo_freqs, octaves = (
            self._config,
            self._modes,
            self._lo_frequencies,
            self._octaves,
        )
        config.set_version()
        if not self.uses_opx_plus():
            config.set_controllers()
        for mode in modes:
            config.set_ports(mode)
            config.set_cores(mode.name, mode.core) # Adrian 10/10/2025
            config.set_upconverters(mode.name, mode.upconverter) # Adrian 10/10/2025
            config.set_intermediate_frequency(mode.name, mode.int_freq)

            if mode.has_mixed_inputs():
                if mode.octave_mixed:
                    config.set_octave_settings(
                        mode.lo_name, self._octaves[mode.lo_name].settings
                    )
                else:
                    if mode.lo_name not in lo_freqs:
                        message = f"No LO frequency specified for {mode = }."
                        raise QMConfigBuildingError(message)
                    lo_freq = lo_freqs[mode.lo_name]
                    config.set_lo_frequency(mode.name, lo_freq)
                    config.set_mixer(mode, mode.int_freq, lo_freq)

            if isinstance(mode, Readout):
                config.set_time_of_flight(mode.name, mode.tof)
                config.set_smearing(mode.name, mode.smearing)

            config.set_operations(mode)

        if self.uses_opx1000():
            config.set_fem_types()
            config = deep_merge(config, self._opx.settings)
            config.infer_mw_fem_settings()
            
        self._config = json.loads(json.dumps(self._config))  # convert recursively to `dict` from `defaultdict`

    def _check_modes(self, *modes: Mode) -> None:
        """ """
        mode_names = []
        for mode in modes:
            if not isinstance(mode, Mode):
                message = f"Invalid {mode = }, must be of {Mode}."
                raise QMConfigBuildingError(message)
            name = mode.name
            if name in mode_names:
                message = f"Found duplicate mode {name = }, names must be unique."
                raise QMConfigBuildingError(message)
            mode_names.append(name)
        self._modes = modes

    def _check_local_oscillators(self, *los: LO) -> None:
        """ """
        for lo in los:
            # todo: currently, since los are remote objects, there is no way to
            #  differentiate between a LMS and an Octave. We will assume for now
            #  that an octave has "octave" in its name.
            if "oct" in lo.name:
                self._octaves[lo.name] = lo
            # try:
            #     self._lo_frequencies[lo.name] = lo.frequency
            # except AttributeError:
            #     message = f"Invalid {lo = }, missing 'name' and 'frequency' attributes."
            #     raise QMConfigBuildingError(message) from None

    def uses_opx_plus(self) -> bool:
        return self._opx.type == 'opx_plus'

    def uses_opx1000(self) -> bool:
        return self._opx.type == 'opx1000'


def deep_merge(d1: QMConfig, d2: dict):
    result = d1  # shallow copy of d1
    for k, v in d2.items():
        if (
            k in result
            and isinstance(result[k], QMConfig)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result