""" """

from qcore.elements.element import Element
from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.pulses.ramped_constant_pulse import ConstantPulse
from qcore.pulses.readout_pulse import ConstantReadoutPulse
from qm import qua
from qcore.helpers.logger import logger


class Readout(Element):
    """ """

    PORTS_KEYS = (*Element.PORTS_KEYS, "out")
    OFFSETS_KEYS = (*Element.OFFSETS_KEYS, "out")

    DEMOD_METHOD_MAP = {
        "sliced": qua.demod.sliced,
        "accumulated": qua.demod.accumulated,
        "window": qua.demod.moving_window,
    }

    def __init__(
        self, time_of_flight: int = 180, smearing: int = 0, **parameters
    ) -> None:
        """ """
        self.time_of_flight: int = time_of_flight
        self.smearing: int = smearing

        if "operations" not in parameters:
            default_operations = [
                ConstantPulse("constant_pulse"),
                ConstantReadoutPulse(
                    "readout_pulse", digital_marker=DigitalWaveform("ADC_ON")
                ),
            ]
            parameters["operations"] = default_operations

        super().__init__(**parameters)

    def measure(
        self,
        targets: tuple,
        ampx=1.0,
        stream: str = None,
        demod_type: str = "full",
        demod_args: tuple = None,
    ) -> None:
        """ """
        iw_key_i, iw_key_q = "cosine", "sine"  # TODO relax hard coding
        var_i, var_q = targets

        if demod_type == "full":
            output_i, output_q = (iw_key_i, var_i), (iw_key_q, var_q)
            demod_i, demod_q = qua.demod.full(*output_i), qua.demod.full(*output_q)
        else:
            try:
                demod_method = self.DEMOD_METHOD_MAP[demod_type]
            except KeyError:
                logger.error(f"Unrecognized demod type '{demod_type}'")
                raise
            else:
                output_i = (iw_key_i, var_i, *demod_args)
                output_q = (iw_key_q, var_q, *demod_args)
                demod_i, demod_q = demod_method(*output_i), demod_method(*output_q)

        key = "readout_pulse"  # TODO relax hard coding
        try:
            qua.measure(key * qua.amp(ampx), self.name, stream, demod_i, demod_q)
        except Exception:  # QM forced me to catch base class Exception...
            try:
                qua.measure(key * qua.amp(*ampx), self.name, stream, demod_i, demod_q)
            except Exception:  # QM forced me to catch base class Exception...
                logger.error("Invalid ampx, expect 1 value or sequence of 4 values")
                raise
