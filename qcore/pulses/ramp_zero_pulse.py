""" """

import numpy as np

from qcore.pulses.pulse import Pulse


class RampZeroPulse(Pulse):
    """
    The flux line pulse with a ramp 0 -> I_ampx, then stable at I_ampx, then ramp again I_ampx -> 0.
    If self.Q_ampx is specified, the q_samples are just scaled with ratio self.I_ampx / self.Q_ampx.
    Args:
    ____________________
    step_V (float): step of the ramp (in Volts). Defaults to 5e-4.
    step_t (int): length of each step of the ramp (in ns). Defaults to 15.
    plateau (int): length of the flat part of the pulse.
    """

    def __init__(
        self,
        name: str,
        step_V: float = 5e-4,
        step_t: int = 15,
        plateau: int = 3500,
        I_ampx: float = 1.0,
        Q_ampx: float = 0.0,  # this is the 'drag' parameter
        pad: int = 0,
        **parameters,
    ) -> None:
        """ """
        self.step_V: float = step_V
        self.step_t: int = step_t
        self.plateau: int = plateau
        super().__init__(
            name=name, length=None, I_ampx=I_ampx, Q_ampx=Q_ampx, pad=pad, **parameters
        )
        del self.length  # not needed once total_length is overriden below

    @property
    def total_length(self) -> int:
        """ """
        return int(
            self.step_t * 2 * self.I_ampx / self.step_V + self.plateau + self.pad
        )

    @property
    def total_I_ampx(self) -> float:
        """ """
        return Pulse.BASE_AMP * self.I_ampx

    def sample(self):
        """ """
        pad = np.zeros(self.pad) if self.pad else []

        rise_ = []
        for _step in np.arange(int(self.I_ampx / self.step_V)):
            rise_.append([_step * self.step_V] * self.step_t)
        rise_ = np.array(rise_).flatten()
        i_samples = (
            np.concatenate([rise_, np.ones(self.plateau) * self.I_ampx, np.flip(rise_)])
            * Pulse.BASE_AMP
        )
        i_wave = (np.concatenate((i_samples, pad))).tolist()

        if self.Q_ampx is None:
            return (i_wave, None)
        elif self.Q_ampx == 0:
            return (i_wave, self.Q_ampx)
        else:
            q_samples = i_samples * self.Q_ampx / self.I_ampx
            q_wave = (np.concatenate((q_samples, pad))).tolist()
            return (i_wave, q_wave)
