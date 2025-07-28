""" """
import numpy as np

from qcore.pulses.readout_pulse import ReadoutPulse
from qcore.pulses.pulse import Pulse

# CLEAR Readout Pulse, creates arbitrary waveform 
class ClearReadoutPulse(ReadoutPulse):
    """ """
    
    def __init__(
        self, 
        ringdown1_amp: float = 0.0,
        ringdown1_time: int = 0,
        ringup1_amp: float = 0.0,
        ringup1_time: int = 0,
        ringdown2_amp: float = 0.0,
        ringdown2_time: int = 0,
        ringup2_amp: float = 0.0,
        ringup2_time: int = 0,
        drive_amp: float = 0.0,
        drive_time: int = 0,
        **parameters
    ) -> None:
        """ """
        self.ringdown1_amp = ringdown1_amp
        self.ringdown1_time = ringdown1_time
        self.ringup1_amp = ringup1_amp
        self.ringup1_time = ringup1_time
        self.ringdown2_amp = ringdown2_amp
        self.ringdown2_time = ringdown2_time
        self.ringup2_amp = ringup2_amp
        self.ringup2_time = ringup2_time
        self.drive_amp = drive_amp
        self.drive_time = drive_time
        
        super().__init__(**parameters)

    @property
    def total_I_amp(self) -> float:
        """ """
        return self.I_ampx
    
    def sample(self):
        """ """                    
        def constant_segment(amplitude, duration):
            return np.full(int(duration), (Pulse.BASE_AMP * amplitude* self.total_I_amp))
        
        # Construct the CLEAR pulse
        ringdown1 = constant_segment(self.ringdown1_amp, self.ringdown1_time)
        ringup1 = constant_segment(self.ringup1_amp, self.ringup1_time)
        ringdown2 = constant_segment(self.ringdown2_amp, self.ringdown2_time)
        ringup2 = constant_segment(self.ringup2_amp, self.ringup2_time)
        drive = constant_segment(self.drive_amp, self.drive_time)

        waveform = np.concatenate([ringup1, ringdown1, drive, ringdown2, ringup2])

        i_samples = np.real(waveform)
        pad = np.zeros(self.pad) if self.pad else []

        i_wave = np.concatenate((i_samples, pad))
        return (i_wave.tolist(), 0.0) if self.has_mixed_waveforms() else (i_wave.tolist(), None)
    
    
class DoubleClearReadoutPulse(ReadoutPulse):
    """ """
    
    def __init__(
        self, 
        ringdown1_amp: float = 0.0,
        ringdown1_time: int = 0,
        ringup1_amp: float = 0.0,
        ringup1_time: int = 0,
        ringdown2_amp: float = 0.0,
        ringdown2_time: int = 0,
        ringup2_amp: float = 0.0,
        ringup2_time: int = 0,
        drive_amp: float = 0.0,
        drive_time: int = 0,
        wait_time: int = 0,
        **parameters
    ) -> None:
        """ """
        self.ringdown1_amp = ringdown1_amp
        self.ringdown1_time = ringdown1_time
        self.ringup1_amp = ringup1_amp
        self.ringup1_time = ringup1_time
        self.ringdown2_amp = ringdown2_amp
        self.ringdown2_time = ringdown2_time
        self.ringup2_amp = ringup2_amp
        self.ringup2_time = ringup2_time
        self.drive_amp = drive_amp
        self.drive_time = drive_time
        self.wait_time = wait_time
        
        super().__init__(**parameters)

    @property
    def total_I_amp(self) -> float:
        """ """
        return self.I_ampx
    
    def sample(self):
        """ """                    
        def constant_segment(amplitude, duration):
            return np.full(int(duration), (Pulse.BASE_AMP * amplitude* self.total_I_amp))
        
        # Construct the CLEAR pulse
        ringdown1 = constant_segment(self.ringdown1_amp, self.ringdown1_time)
        ringup1 = constant_segment(self.ringup1_amp, self.ringup1_time)
        ringdown2 = constant_segment(self.ringdown2_amp, self.ringdown2_time)
        ringup2 = constant_segment(self.ringup2_amp, self.ringup2_time)
        drive = constant_segment(self.drive_amp, self.drive_time)
        wait = constant_segment(0.0, self.wait_time)

        waveform = np.concatenate([ringup1, ringdown1, drive, ringdown2, ringup2, wait, ringup1, ringdown1, drive, ringdown2, ringup2])

        i_samples = np.real(waveform)
        pad = np.zeros(self.pad) if self.pad else []

        i_wave = np.concatenate((i_samples, pad))
        return (i_wave.tolist(), 0.0) if self.has_mixed_waveforms() else (i_wave.tolist(), None)