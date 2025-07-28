""" """

from pathlib import Path
from typing import Union

import numpy as np

from qcore.pulses.constant_pulse import ConstantPulse
from qcore.pulses.gaussian_pulse import GaussianPulse
from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.pulses.pulse import Pulse

from qualang_tools.config.integration_weights_tools import convert_integration_weights
DIVISION_LEN = 16

# Integration weight dictionary
IW = dict[str, list]

# cos and sin integration weight tuple
CosSinWeightTuple = tuple[float, float, float, float]
# cos, sin and -sin integration weight tuple (used for dual-demodulation)
CosSinMinusSinWeightTuple = tuple[float, float, float, float, float, float]


class ReadoutPulse(Pulse):
    """ """

    def __init__(
        self,
        name: str,
        weights: Union[CosSinWeightTuple, CosSinMinusSinWeightTuple, str] = (
            1.0,
            0.0,  # cos
            0.0,
            1.0,  # sin
            0.0,
            -1.0,  # minus_sin
        ),
        threshold: Union[float, None] = None,  # not None only for Optimized weights
        **parameters,
    ) -> None:
        """ """
        # for constant Weights, specify tuple (i_cos, i_sin, q_cos, q_sin)
        # for optimized Weights, specify path string to npz file which must have
        # two arrays named "I" and "Q" which store cosine and sine weights respectively
        self.weights = weights
        self.threshold = threshold
        super().__init__(name=name, **parameters)

    @property
    def has_optimized_weights(self) -> bool:
        """ """
        try:
            return Path(self.weights).exists()
        except TypeError:
            return False
    
    def _convert_integration_weights(self, weights):
        opt_weights = convert_integration_weights(weights)

        for i in range(len(opt_weights)):
            opt_weights[i] = (opt_weights[i][0], DIVISION_LEN * opt_weights[i][1])
        
        return opt_weights

    def sample_integration_weights(self) -> tuple[IW, IW, IW]:
        """ """
        minus_sin_weights = {"cosine": 0.0, "sine": -1.0}
        if self.has_optimized_weights:
            weights = np.load(self.weights)
            if "Q_Minus" in weights:
                # required for dual-demodulation
                minus_sin_weights = {
                    "cosine": self._convert_integration_weights(weights["Q_Minus"][0]),
                    "sine": self._convert_integration_weights(weights["Q_Minus"][1]),
                }
                cos_weights = {
                    "cosine":self._convert_integration_weights(weights["I"][0]),
                    "sine": self._convert_integration_weights(weights["I"][1]),
                }
                sin_weights = {
                    "cosine": self._convert_integration_weights(weights["Q"][0]),
                    "sine": self._convert_integration_weights(weights["Q"][1]),
                }
            else:
                cos_weights = {"cosine": weights["I"][0], "sine": weights["I"][1]}
                sin_weights = {"cosine": weights["Q"][0], "sine": weights["Q"][1]}

        else:
            weights = [[(weight, self.total_length)] for weight in self.weights]
            cos_weights = {"cosine": weights[0], "sine": weights[1]}
            sin_weights = {"cosine": weights[2], "sine": weights[3]}
            if len(weights) >= 6:
                # required for dual-demodulation
                minus_sin_weights = {"cosine": weights[4], "sine": weights[5]}

        return cos_weights, sin_weights, minus_sin_weights


class ConstantReadoutPulse(ConstantPulse, ReadoutPulse):
    """ """


class GaussianReadoutPulse(GaussianPulse, ReadoutPulse):
    """ """
    
class DoubleConstantReadoutPulse(ReadoutPulse):
    """ """
    def __init__(
        self,
        name: str,
        length: int = 1000,  # in ns
        I_ampx: float = 1.0,
        Q_ampx: Union[None, float] = 0.0,
        pad: int = 0,
        wait_time: int = 0,
        **parameters,
    ) -> None:
        """ """
        self.wait_time = wait_time
        
        super().__init__(
            name=name,
            length=length,
            I_ampx=I_ampx,
            Q_ampx=Q_ampx,
            pad=pad,
            **parameters,
        )

    @property
    def total_I_amp(self) -> float:
        """ """
        return Pulse.BASE_AMP * self.I_ampx

    def sample(self):
        """ """
        def constant_segment(amplitude, duration):
            return np.full(int(duration), (Pulse.BASE_AMP * amplitude))
        
        # Construct the CLEAR pulse
        const = constant_segment(self.total_I_amp, (self.length-self.wait_time)/2)
        wait = constant_segment(0.0, self.wait_time)

        waveform = np.concatenate([const, wait, const])

        i_samples = np.real(waveform)
        pad = np.zeros(self.pad) if self.pad else []

        i_wave = np.concatenate((i_samples, pad))
        return (i_wave.tolist(), 0.0) if self.has_mixed_waveforms() else (i_wave.tolist(), None)
