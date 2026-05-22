import numpy as np
from qcore.helpers.logger import logger
from qcore.instruments.drivers.qm_config import QMConfig
from qcore.modes import Readout
from qcore.modes.mode import FEMPort, Mode


class QMConfigOPX1000(QMConfig):
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

    def __init__(self, controllers_info: dict = None) -> None:
        self.controllers_info = controllers_info
        self.allocate_frequencies_automatically = False
        super().__init__()

    def set_controllers(self) -> None:
        """ """
        self["controllers"][QMConfig._CONTROLLER_NAME]["type"] = "opx1000"

    def set_analog_output_port(self, port: FEMPort, offset: float) -> None:
        fem, port = port
        self.check_output_port_bounds(port, "Analog output port")
        controllers_config = self["controllers"][QMConfig._CONTROLLER_NAME]["fems"][fem]
        controllers_config["analog_outputs"][port] = type(self)()
        if self.fem_type(self._CONTROLLER_NAME, fem) == "LF":
            controllers_config["analog_outputs"][port]["offset"] = 0.
        logger.debug(f"Set controller analog output port {port = } with {offset = }.")

    def set_analog_input_port(self, port: FEMPort, offset: float) -> None:
        """ """
        fem, port = port
        self.check_input_port_bounds(port, "Analog input port")
        controllers_config = self["controllers"][QMConfig._CONTROLLER_NAME]["fems"][fem]
        controllers_config["analog_inputs"][port]["gain_db"] = 0
        if self.fem_type(self._CONTROLLER_NAME, fem) == "LF":
            controllers_config["analog_inputs"][port]["offset"] = 0.
        logger.debug(f"Set controller analog input port {port = } with {offset = }.")

    def set_digital_output_port(self, port: FEMPort) -> None:
        """ """
        fem, port = port
        self.check_output_port_bounds(port, "Digital output port")
        controllers_config = self["controllers"][QMConfig._CONTROLLER_NAME]["fems"][fem]
        controllers_config["digital_outputs"][port] = {}
        logger.debug(f"Set controller digital output port {port = }.")

    def set_digital_input_config(self, digital_input_config: dict, mode: Mode) -> None:
        digital_input_config["port"] = (self._CONTROLLER_NAME, *mode.rf_switch.port)
        digital_input_config["buffer"] = mode.rf_switch.buffer
        digital_input_config["delay"] = mode.rf_switch.delay

    def set_fem_types(self):
        for controller_name, fems in self.controllers_info.items():
            for fem, fem_type in fems.items():
                self["controllers"][controller_name]["fems"][fem]["type"] = fem_type

    def fem_type(self, controller_name: str, fem: int) -> str:
        return self.controllers_info[controller_name][fem]

    def set_mode_port(self, mode: Mode, key: str, port: FEMPort) -> None:
        """ """
        if key not in Readout.PORTS_KEYS:
            raise ValueError(f"Invalid port {key = }, must be in {Readout.PORTS_KEYS}.")

        mode_config = self["elements"][mode.name]
        port_config = (self._CONTROLLER_NAME, *port)
        fem, port = port

        if "out" in key:
            if mode.octave_mixed:
                self.set_mode_output_port_octave(mode, port)
            else:
                if self.fem_type(self._CONTROLLER_NAME, fem) == "MW":
                    mode_config["MWOutput"]["port"] = port_config
                else:
                    mode_config["outputs"][key] = port_config
        elif key in ("I", "Q") and mode.has_mixed_inputs():
            if mode.octave_mixed:
                self.set_mode_input_port_octave(mode, (fem, port))
            else:
                mode_config["mixInputs"][key] = port_config
        elif key == "I":
            if self.fem_type(self._CONTROLLER_NAME, fem) == "MW":
                mode_config["MWInput"]["port"] = port_config
            else:
                mode_config["singleInput"]["port"] = port_config
        else:
            raise ValueError(f"Invalid port {key = } and {port = } for {mode}.")
        logger.debug(f"Set '{mode.name}' port {key = } and {port = }.")

        if mode.rf_switch_on:
            rf_switch_config = mode_config["digitalInputs"]["switch"]
            self.set_digital_input_config(rf_switch_config, mode)


    def infer_mw_fem_settings(self):
        if self.allocate_frequencies_automatically:
            self.set_ducs()
        else:
            self._set_upconverters()
        self.set_bands()

    def _set_upconverters(self):
        """
        By default, we assume that we only ever have one upconverter, so we set
        all element upconverters to index 1.
        """
        for element in self["elements"].values():
            if "MWInput" in element:
                element["MWInput"]["upconverter"] = element["upconverter"]
                del element["upconverter"]

    def set_bands(self):
        """
        Set the frequency band of a port according to the digital upconverter frequencies
        on the port.
        """
        def get_band(frequency):
            if 4.5e9 <= frequency <= 7.5e9:
                return 2
            elif 50e6 <= frequency <= 5.5e9:
                return 1
            elif 6.5e9 <= frequency <= 10.5e9:
                return 3
            else:
                raise ValueError(f"Expected frequency inside range [50MHz, 10.5GHz], got {frequency}")

        # set MW-FEM bands, assuming always using band 2
        for controller in self["controllers"].values():
            for fem in controller["fems"].values():
                # print(fem) # used to find out which port is being used
                if fem["type"] == "MW":
                    for analog_output in fem["analog_outputs"].values():

                        print(analog_output)
                        analog_output["band"] = get_band(analog_output["upconverters"][1]["frequency"])

                         # Band 2 has a 20 ns delay compared to band 1 and 3               # Jon, 08/09/25
                        if analog_output["band"] == 1 or analog_output["band"] == 3:
                            analog_output["delay"] = 20
                        
                    for analog_input in fem["analog_inputs"].values():
                        analog_input["band"] = get_band(analog_input["downconverter_frequency"])


    def set_ducs(self):
        """
        Translate the desired frequency for each element to a corresponding digital
        up/downconverter frequency plus intermediate frequency. This is done carefully,
        due to the following constraints:
         - Intermediate frequencies can span no further than ±500MHz from a DUC frequency
         - There are only 4 DUCs allowed per coupler pair of ports (so, choose sparingly)
         - Readout cannot be performed within 10MHz of a DUC frequency

        """
        # Determine the set of frequencies required on a particular port.
        frequencies_by_port = QMConfig()
        for element in self["elements"].values():
            for element_type in ["MWInput", "MWOutput"]:
                if element_type in element:
                    port = element[element_type]["port"]
                    frequencies_config = frequencies_by_port[port[0]][port[1]][port[2]]
                    frequencies_config[len(frequencies_config)] = element["intermediate_frequency"]

        # For each port:
        for fems in frequencies_by_port.values():
            grouped_frequencies_by_port = {}
            for fem, ports in fems.items():
                for port, freqs in ports.items():
                    # Split frequencies into smallest subsets, spanning no more than 1GHz
                    subsets = []
                    current_subset = []
                    freqs = sorted(list(freqs.values()))
                    for freq in freqs:
                        if not current_subset:
                            current_subset.append(freq)
                        elif max(current_subset + [freq]) - min(current_subset + [freq]) <= 1e9:
                            current_subset.append(freq)
                        else:
                            subsets.append(current_subset)
                            current_subset = [freq]

                    if current_subset:
                        subsets.append(current_subset)

                    if len(subsets) > 4:
                        raise ValueError(f"Too many DUCs needed for {fem = }, {port = }: {subsets = }")

                    grouped_frequencies_by_port[port] = subsets

                    converter_frequencies = []
                    for group in grouped_frequencies_by_port[port]:
                        group_center = (min(group) + max(group)) / 2

                        # Use the center of the frequency subset if it's valid
                        center_is_valid = True
                        for frequency in group:
                            if abs(frequency - group_center) < 20e6:
                                center_is_valid = False
                                break

                        if center_is_valid:
                            converter_frequencies.append(group_center)

                        else:
                            # Otherwise, find allowed regions
                            sorted_group = sorted(group)

                            # Create exclusion zones [freq - 20MHz, freq + 20MHz]
                            exclusion_zones = [(freq - 20e6, freq + 20e6) for freq in sorted_group]

                            # Merge overlapping exclusion zones
                            merged_zones = []
                            for start, end in sorted(exclusion_zones):
                                if not merged_zones or merged_zones[-1][1] < start:
                                    merged_zones.append([start, end])
                                else:
                                    merged_zones[-1][1] = max(merged_zones[-1][1], end)

                            # Build allowed zones between exclusions
                            allowed_zones = []
                            last_end = -float('inf')
                            for start, end in merged_zones:
                                if last_end != -float('inf') and last_end < start:
                                    allowed_zones.append((last_end, start))
                                last_end = end
                            allowed_zones.append((last_end, float('inf')))

                            # Find best candidate: either center (if inside allowed zone) or nearest boundary
                            best_candidate = None
                            best_distance = float('inf')
                            for start, end in allowed_zones:
                                if start <= group_center <= end:
                                    best_candidate = group_center
                                    break
                                else:
                                    # Try the boundaries
                                    for boundary in (start, end):
                                        distance = abs(boundary - group_center)
                                        if distance < best_distance:
                                            best_distance = distance
                                            best_candidate = boundary

                            converter_frequencies.append(best_candidate)

                    ports[port] = converter_frequencies

        # create a DUC for each subset in the controllers, i.e.,
        #   "upconverters": {1: ..., 2: ...},
        for controller_name, controller in frequencies_by_port.items():
            for fem_index, fem in controller.items():
                for port_index, port in fem.items():
                    for duc_index, duc_frequency in enumerate(port):
                        port_config = self["controllers"][controller_name]["fems"][fem_index]
                        if "analog_outputs" in port_config and port_index in port_config["analog_outputs"]:
                            port_config["analog_outputs"][port_index]["upconverters"][duc_index + 1] = {
                                "frequency": duc_frequency
                            }
                        if "analog_inputs" in port_config and port_index in port_config["analog_inputs"]:
                            if len(port) > 1:
                                raise NotImplementedError(
                                    f"Since multiple downconverter frequencies are not supported in the "
                                    f"OPX1000, it is unclear which one to use when multiple DUCs are "
                                    f"present. ")
                            port_config["analog_inputs"][port_index]["downconverter_frequency"] = duc_frequency

        # make sure each element points to an upconverter/downconverter
        for element in self["elements"].values():
            if "MWInput" in element:
                port = element["MWInput"]["port"]
                current_if = element["intermediate_frequency"]
                upconverter_frequencies = np.array(frequencies_by_port[port[0]][port[1]][port[2]])
                best_upconverter = int(abs(upconverter_frequencies - current_if).argmin())
                element["MWInput"]["upconverter"] = best_upconverter + 1
                element["intermediate_frequency"] = int(current_if - upconverter_frequencies[best_upconverter])
