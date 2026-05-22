""" """

import numpy as np

from qcore.pulses.constant_pulse import ConstantPulse


def ramp_cos(length: int, up: bool = True) -> list[float]:
    """ """
    samples = 0.5 * (1 - np.cos(np.linspace(0, np.pi, length)))
    return samples if up else samples[::-1]

def ramp_cos2(length: int, up: bool = True) -> list[float]:
    """ """
    samples = 0.25 * (1 - np.cos(np.linspace(0, np.pi, length)))**2
    return samples if up else samples[::-1]

def ramp_tanh(length: int, up: bool = True) -> list[float]:
    """ """
    samples = (1 + np.tanh(np.linspace(-2, 2, length))) / 2
    return samples if up else samples[::-1]


RAMP_MAP = {"cos": ramp_cos, "cos2": ramp_cos2, "tanh": ramp_tanh}


class RampUpPulse(ConstantPulse):
    """ """

    def __init__(
        self,
        name: str,
        ramp: int = 0,
        rampfn: str = "cos",
        **parameters,
    ) -> None:
        """ """
        self.ramp: int = ramp
        self.rampfn: str = rampfn
        super().__init__(name, **parameters)

    @property
    def total_length(self) -> None:
        """ """
        return self.ramp

    def sample(self):
        """ """
        has_constant_waveform = not (self.pad or self.ramp)
        if has_constant_waveform:
            return self._sample_constant_waveform()
        else:
            return self._sample_arbitrary_waveform()

    def _sample_arbitrary_waveform(self):
        rampfn = RAMP_MAP[self.rampfn] if self.rampfn is not None else False
        up = rampfn(self.ramp, up=True) if rampfn else []
        i_wave = (up * self.total_I_amp).tolist()

        return (i_wave, 0.0) if self.has_mixed_waveforms() else (i_wave, None)
