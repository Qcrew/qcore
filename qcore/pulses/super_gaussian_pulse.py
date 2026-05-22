""" """
import numpy as np
from qcore.pulses.pulse import Pulse

class SuperGaussianPulse(Pulse):
    """ """

    def __init__(
        self,
        name: str,
        sigma: float = 10,
        chop: int = 6,
        I_ampx: float = 1.0,
        Q_ampx: float = 0.0,  # this is the 'drag' parameter
        pad: int = 0,
        **parameters,
    ) -> None:
        """ """
        self.sigma: float = sigma
        self.chop: int = chop
        super().__init__(name=name, length=None, I_ampx=I_ampx, Q_ampx=Q_ampx, pad=pad, **parameters)
        del self.length  # not needed once total_length is overriden below

    @property
    def total_length(self) -> int:
        """ """
        return int(self.sigma * self.chop) + self.pad

    @property
    def total_I_ampx(self) -> float:
        """ """
        return Pulse.BASE_AMP * self.I_ampx

    def sample(self):
        """ """
        start, stop = -self.chop / 2 * self.sigma, self.chop / 2 * self.sigma
        length = int(self.sigma * self.chop)
        ts = np.linspace(start, stop, length)
        pad = np.zeros(self.pad) if self.pad else []

        i_samples = np.exp(-(ts**10) / (2.0 * self.sigma**10)) *  self.total_I_ampx
        i_wave = (np.concatenate((i_samples, pad))).tolist()

        if self.Q_ampx is None:
            return (i_wave, None)
        elif self.Q_ampx == 0:
            return (i_wave, self.Q_ampx)
        else:
            q_samples = (np.exp(0.5) / self.sigma) * -ts * i_samples * self.Q_ampx
            q_wave = (np.concatenate((q_samples, pad))).tolist()
            return (i_wave, q_wave)
