import math
import numpy as np
from collections import defaultdict
from typing import Any, Union, List

from qcore.instruments.drivers.qm_port import Port
from qcore.modes.rf_switch import RFSwitch
from qcore.helpers.logger import logger
from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.pulses.pulse import Pulse
from qcore.pulses.readout_pulse import ReadoutPulse
from qcore.modes.mode import Mode, FEMPort
from qcore.modes.readout import Readout


class QMConfig(defaultdict):
    """https://qm-docs.qualang.io/introduction/config"""

    _CONTROLLER_NAME: str = "con1"
    _MIN_OUTPUT_PORTS: int = 1
    _MAX_OUTPUT_PORTS: int = 10
    _MIN_INPUT_PORTS: int = 1
    _MAX_INPUT_PORTS: int = 2
    _MIN_WAVEFORM_VOLTAGE: float = -0.5  # V
    _MAX_WAVEFORM_VOLTAGE: float = 0.5
    _MIN_MCM_VALUE: float = -2.0  # MCM means mixer correction matrix
    _MAX_MCM_VALUE: float = 2 - 2**-16
    _CLOCK_CYCLE: int = 4  # ns, also defined in qcore.pulses.pulse.Pulse
    _MIN_TIME_OF_FLIGHT: int = 24  # ns
    _MIN_PULSE_LENGTH: int = 16  # ns
    _MAX_PULSE_LENGTH: int = 2**31 - 1  # ns

    def __init__(self) -> None:
        """ """
        super().__init__(QMConfig)

    def __repr__(self) -> str:
        """ """
        return repr(dict(self))

    def set_version(self) -> None:
        """ """
        self["version"] = 1

    def set_ports(self, mode: Mode) -> None:
        ports = mode.ports
        for port_key, port in ports.items():
            if port is not None:
                self.set_controller_port(mode, port_key, port)
                self.set_mode_port(mode, port_key, port)
 
    def set_cores(self, name: str, value: str) -> None: # Adrian 10/10/2025
        if value is not None:
            self["elements"][name]["core"] = self.cast(value, str, "core")

    def set_upconverters(self, name: str, value: int) -> None: # Kyle 13/03/2026
        if value is not None:
            self["elements"][name]["upconverter"] = self.cast(value, int, "upconverter")
        
    def set_intermediate_frequency(self, name: str, value: float) -> None:
        """ """
        int_freq = self.cast(value, int, "intermediate frequency")
        self["elements"][name]["intermediate_frequency"] = int_freq
        logger.debug(f"Set {name} {int_freq = }.")

    def set_lo_frequency(self, name: str, value: float) -> None:
        """ """
        lo_freq = self.cast(value, int, "lo frequency")
        self["elements"][name]["mixInputs"]["lo_frequency"] = lo_freq
        logger.debug(f"Set {name} {lo_freq = }.")

    def set_mixer(self, mode: Mode, int_freq: float, lo_freq: float) -> None:
        """ """
        ports, offsets = mode.ports, mode.mixer_offsets
        mixer_name = f"mixer_{ports['I']}{ports['Q']}"

        mixer_correction_matrix = self.get_correction_matrix(offsets["G"], offsets["P"])
        self.check_mcm_bounds(mixer_correction_matrix)
        mixer_config = {
            "intermediate_frequency": int(int_freq),
            "lo_frequency": int(lo_freq),
            "correction": mixer_correction_matrix,
        }
        if mixer_name in self["mixers"]:
            self["mixers"][mixer_name].append(mixer_config)
        else:
            self["mixers"][mixer_name] = [mixer_config]

        self["elements"][mode.name]["mixInputs"]["mixer"] = mixer_name
        logger.debug(f"Set {mode} {mixer_config = }.")

    def set_octave_settings(self, octave_name: str, settings: dict) -> None:
        if "octaves" not in self:
            self["octaves"] = {}
        self["octaves"][octave_name] = settings

    def set_time_of_flight(self, name: str, value: int) -> None:
        """ """
        self.check_tof_bounds(value)
        tof, cycle = int(value), QMConfig._CLOCK_CYCLE
        if tof % cycle != 0:
            tof = cycle * round(tof / cycle)
            logger.warning(f"{name} time of flight rounded to multiple of {cycle}.")

        self["elements"][name]["time_of_flight"] = tof
        logger.debug(f"Set {name} time of flight to {tof}.")

    def set_smearing(self, name: str, value: int) -> None:
        """ """
        smearing = self.cast(value, int, "smearing")
        self["elements"][name]["smearing"] = smearing
        logger.debug(f"Set {name} {smearing = }.")

    def set_operations(self, mode: Mode) -> None:
        """ """
        for op_name, pulse in mode.operations.items():
            pulse_name = mode.name + "." + pulse.name
            self["elements"][mode.name]["operations"][op_name] = pulse_name
            self.set_pulse(pulse, pulse_name)

    def cast(self, value: Any, cls: Any, key: str) -> Any:
        """ """
        try:
            return cls(value)
        except (TypeError, ValueError):
            raise ValueError(f"Failed to cast {key} {value = } to {cls}, invalid type.")

    def check_bounds(self, value: float, min: float, max: float, key: str) -> None:
        """ """
        try:
            in_bounds = min <= value <= max
        except TypeError:
            message = f"Invalid type for {key} {value = }, must be a number"
            raise ValueError(message) from None
        else:
            if not in_bounds:
                raise ValueError(f"{key} {value = } out of bounds: [{min}, {max}].")

    def check_voltage_bounds(self, value: float, key: str) -> None:
        """ """
        min, max = self._MIN_WAVEFORM_VOLTAGE, self._MAX_WAVEFORM_VOLTAGE
        self.check_bounds(value, min, max, key)

    def check_output_port_bounds(self, value: int, key: str) -> None:
        """ """
        min, max = self._MIN_OUTPUT_PORTS, self._MAX_OUTPUT_PORTS
        self.check_bounds(value, min, max, key)

    def check_input_port_bounds(self, value: int, key: str) -> None:
        """ """
        min, max = self._MIN_INPUT_PORTS, self._MAX_INPUT_PORTS
        self.check_bounds(value, min, max, key)

    def check_mcm_bounds(self, mcm: tuple[float, float, float, float]) -> None:
        """ """
        min, max = self._MIN_MCM_VALUE, self._MAX_MCM_VALUE
        for value in mcm:
            self.check_bounds(value, min, max, "Mixer correction matrix")

    def check_tof_bounds(self, value: int) -> None:
        """ """
        min, max = self._MIN_TIME_OF_FLIGHT, np.inf
        self.check_bounds(value, min, max, "Time of flight")

    def check_pulse_length(self, value: int, key: str) -> None:
        """ """
        min, max = self._MIN_PULSE_LENGTH, self._MAX_PULSE_LENGTH
        self.check_bounds(value, min, max, key)

        clock_cycle = self._CLOCK_CYCLE
        if value % clock_cycle != 0:
            message = f"'{key}' length = {value} must be a multiple of {clock_cycle}"
            raise ValueError(message)

    def set_controllers(self) -> None:
        """ """
        self["controllers"][QMConfig._CONTROLLER_NAME]["type"] = "opx1"

    def set_controller_port(self, mode: Mode, key: str, port: Port) -> None:
        """ """
        dc_offset = mode.mixer_offsets[key]
        self.check_voltage_bounds(dc_offset, "DC offset voltage")
        if key in ("I", "Q"):
            self.set_analog_output_port(port, dc_offset)
        elif "out" in key:
            self.set_analog_input_port(port, dc_offset)
        else:
            raise ValueError(f"Invalid port {key = }, must be in ('I', 'Q', 'out').")

        rf_switch = mode.rf_switch
        if rf_switch is not None:
            self.set_digital_output_port(rf_switch.port)

    def set_analog_output_port(self, port: int, offset: float) -> None:
        """ """
        self.check_output_port_bounds(port, "Analog output port")
        controllers_config = self["controllers"][self._CONTROLLER_NAME]
        controllers_config["analog_outputs"][port]["offset"] = offset
        logger.debug(f"Set controller analog output port {port = } with {offset = }.")

    def set_analog_input_port(self, port: int, offset: float) -> None:
        """ """
        self.check_input_port_bounds(port, "Analog input port")
        controllers_config = self["controllers"][self._CONTROLLER_NAME]
        controllers_config["analog_inputs"][port]["offset"] = offset
        controllers_config["analog_inputs"][port]["gain_db"] = 0
        logger.debug(f"Set controller analog input port {port = } with {offset = }.")

    def set_digital_output_port(self, port: Port) -> None:
        """ """
        if isinstance(port, List):
            fem, port = port
        self.check_output_port_bounds(port, "Digital output port")
        controllers_config = self["controllers"][self._CONTROLLER_NAME]
        controllers_config["digital_outputs"][port] = {}
        logger.debug(f"Set controller digital output port {port = }.")

    def set_mode_port(self, mode: Mode, key: str, number: int) -> None:
        """ """
        if key not in Readout.PORTS_KEYS:
            raise ValueError(f"Invalid port {key = }, must be in {Readout.PORTS_KEYS}.")

        mode_config = self["elements"][mode.name]
        port_config = (self._CONTROLLER_NAME, number)
        if "out" in key:
            if mode.octave_mixed:
                self.set_mode_output_port_octave(mode, number)
            else:
                mode_config["outputs"][key + str(number)] = port_config
        elif key in ("I", "Q") and mode.has_mixed_inputs():
            if mode.octave_mixed:
                self.set_mode_input_port_octave(mode, number)
            else:
                mode_config["mixInputs"][key] = port_config
        elif key == "I":
            mode_config["singleInput"]["port"] = port_config
        else:
            raise ValueError(f"Invalid port {key = } and {number = } for {mode}.")
        logger.debug(f"Set '{mode.name}' port {key = } and {number = }.")

        if mode.rf_switch_on:
            rf_switch_config = mode_config["digitalInputs"]["switch"]
            self.set_digital_input_config(rf_switch_config, mode)

    def set_digital_input_config(self, digital_input_config: dict, mode: Mode) -> None:
        digital_input_config["port"] = (self._CONTROLLER_NAME, mode.rf_switch.port)
        digital_input_config["buffer"] = mode.rf_switch.buffer
        digital_input_config["delay"] = mode.rf_switch.delay

    def set_mode_output_port_octave(self, mode: Mode, port: Port):
        if isinstance(port, List):
            port = port[-1]
        self["elements"][mode.name]["RF_outputs"]["port"] = (
            mode.lo_name,
            math.ceil(port / 2),
        )

    def set_mode_input_port_octave(self, mode: Mode, port: Port):
        # OPX->Octave standard connectivity assumes:
        #   OPX 1,2 = I,Q -> Octave 1
        #   OPX 3,4 = I,Q -> Octave 2
        #   ... etc
        if isinstance(port, List):
            port = port[-1]
        self["elements"][mode.name]["RF_inputs"]["port"] = (
            mode.lo_name,
            math.ceil(port / 2),
        )

    def get_correction_matrix(self, g: float, p: float) -> list[float]:
        """ """
        try:
            cos, sin = np.cos(p), np.sin(p)
            coefficient = 1 / ((1 - g**2) * (2 * cos**2 - 1))
        except TypeError:
            message = f"Invalid offset value(s): {g = }, {p = }, both must be {float}."
            raise ValueError(message) from None
        else:
            matrix = ((1 - g) * cos, (1 + g) * sin, (1 - g) * sin, (1 + g) * cos)
            return [coefficient * value for value in matrix]

    def set_pulse(self, pulse: Pulse, pulse_name: str) -> None:
        """ """
        pulse_config = self["pulses"][pulse_name]
        pulse_type = "measurement" if isinstance(pulse, ReadoutPulse) else "control"
        pulse_config["operation"] = pulse_type
        self.set_pulse_length(pulse_name, pulse.total_length)

        if pulse.has_mixed_waveforms():
            waveform_I_name = pulse_name + ".waveform." + "I"
            waveform_Q_name = pulse_name + ".waveform." + "Q"
            pulse_config["waveforms"]["I"] = waveform_I_name
            pulse_config["waveforms"]["Q"] = waveform_Q_name
            self.set_waveforms(pulse, waveform_I_name, waveform_Q_name)
        else:
            waveform_name = pulse_name + ".waveform"
            pulse_config["waveforms"]["single"] = waveform_name
            self.set_waveforms(pulse, waveform_name)

        digital_marker = pulse.digital_marker
        if digital_marker is not None:
            marker_name = pulse_name + "." + digital_marker.name
            pulse_config["digital_marker"] = marker_name
            self.set_digital_waveform(digital_marker, marker_name)

        if pulse_type == "measurement" and pulse.has_mixed_waveforms():
            iw_cos_name, iw_sin_name, iw_minus_sin_name = (
                pulse_name + ".cos",
                pulse_name + ".sin",
                pulse_name + ".minus_sin",
            )
            pulse_config["integration_weights"]["cos"] = iw_cos_name
            pulse_config["integration_weights"]["sin"] = iw_sin_name
            pulse_config["integration_weights"]["minus_sin"] = iw_minus_sin_name
            self.set_integration_weights(
                pulse, iw_cos_name, iw_sin_name, iw_minus_sin_name
            )

    def set_pulse_length(self, name: str, value: int) -> None:
        """ """
        length = self.cast(value, int, "pulse length")
        self.check_pulse_length(value, f"Pulse '{name}' length")
        self["pulses"][name]["length"] = length
        logger.debug(f"Set '{name}' {length = }.")

    def set_waveforms(self, pulse: Pulse, wf_i: str, wf_q: str = None) -> None:
        """ """
        i_wave, q_wave = pulse.sample()
        waveform_dict = {wf_i: i_wave, wf_q: q_wave}
        for name, wave in waveform_dict.items():
            if wave is not None:
                try:
                    wave_len = len(wave)
                except TypeError:
                    waveform_type = "constant"
                else:
                    waveform_type = "arbitrary"
                    pulse_len = pulse.total_length
                    if not pulse_len == wave_len:
                        message = f"Unequal '{name}' {wave_len = } and {pulse_len = }."
                        raise ValueError(message)

                self.set_waveform(name, waveform_type, wave)

    def set_waveform(self, name: str, type: str, sample) -> None:
        """ """
        self["waveforms"][name]["type"] = type
        if type == "constant":
            self.set_constant_waveform(name, sample)
        elif type == "arbitrary":
            self.set_arbitrary_waveform(name, sample)

    def set_constant_waveform(self, name: str, sample: float) -> None:
        """ """
        self.check_voltage_bounds(sample, f"'{name}' voltage")
        self["waveforms"][name]["sample"] = sample
        logger.debug(f"Set constant waveform '{name}' with {sample = }.")

    def set_arbitrary_waveform(self, name: str, samples: list[float]) -> None:
        """ """
        self.check_voltage_bounds(min(samples), f"'{name}' voltage")
        self.check_voltage_bounds(max(samples), f"'{name}' voltage")
        self["waveforms"][name]["samples"] = samples
        self["waveforms"][name]["max_allowed_error"] = 1   ### ADRIAN 7th October 2025

        logger.debug(f"Set arbitrary waveform '{name}' with {len(samples)} samples.")

    def set_digital_waveform(self, waveform: DigitalWaveform, name: str) -> None:
        """ """
        self["digital_waveforms"][name]["samples"] = waveform.samples
        logger.debug(f"Set digital waveform '{name}'.")

    def set_integration_weights(
            self, pulse: ReadoutPulse, cos: str, sin: str, minus_sin: str = None
    ) -> None:
        """ """
        cos_weights, sin_weights, minus_sin_weights = pulse.sample_integration_weights()
        self["integration_weights"][cos] = cos_weights
        self["integration_weights"][sin] = sin_weights
        if minus_sin is not None:
            self["integration_weights"][minus_sin] = minus_sin_weights

    def set_fem_types(self):
        pass
