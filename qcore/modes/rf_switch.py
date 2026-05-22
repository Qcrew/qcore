""" """
from qualang_tools.config import Port

from qcore.resource import Resource


class RFSwitch(Resource):
    """ """

    # the OPX has an intrinsic delay of analog channel with respect to digital channel
    INTRINSIC_DELAY: int = 136

    def __init__(self, name: str, port: tuple, delay: int = 0, buffer: int = 0, intrinsic_delay: bool = False) -> None:
        """ """
        self.port = port  # OPX digital output port this switch is connected to

        # e.g. if digital waveform sample = [(1, 0)], delay = 10, buffer = 4
        # the digital output starts 10 + 4 = 14ns after its associated analog output
        # and is high for 2 * 4 = 8ns longer than its associated analog output
        if intrinsic_delay:
            self.delay = delay + RFSwitch.INTRINSIC_DELAY
        else:
            self.delay = delay
        self.buffer = buffer  # defines broadening of the digital signal
        super().__init__(name=name)
