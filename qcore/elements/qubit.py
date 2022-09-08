""" """

from qcore.elements.mode import Mode
from qcore.pulses.ramped_constant_pulse import ConstantPulse
from qcore.pulses.gaussian_pulse import GaussianPulse

class Qubit(Mode):
    """ """

    def __init__(self, **parameters) -> None:
        """ """
        if "ops" not in parameters:
            default_ops = {
                "constant_pulse": ConstantPulse(),
                "gaussian_pulse": GaussianPulse(),
            }
            parameters["ops"] = default_ops

        super().__init__(**parameters)

    def rotate(self) -> None:
        """ """