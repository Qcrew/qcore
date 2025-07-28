""" """

from qcore.resource import Resource


class RFSwitch(Resource):
    """ """

    # IMPORTANT: PLEASE READ ME
    # The OPX has an intrinsic delay of analog channel with respect to digital channel of 136ns
    # this means that the digital pulse starts 136ns *before* the analog pulse. The delay param
    # tells the OPX to delay the digital pulse which shortens the time difference between the 2
    # pulses. To calculate the time between both pulses, use the formula 136 - delay + buffer.
    # Note that the buffer cannot be greater than the delay.

    def __init__(self, name: str, port: int, delay: int = 0, buffer: int = 0) -> None:
        """ """
        self.port = port  # OPX digital output port this switch is connected to

        # e.g. if digital waveform sample = [(1, 0)], delay = 10, buffer = 4
        # the digital output starts 10 + 4 = 14ns after its associated analog output
        # and is high for 2 * 4 = 8ns longer than its associated analog output
        self.delay = delay
        self.buffer = buffer  # defines broadening of the digital signal
        super().__init__(name=name)
