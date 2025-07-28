from qm.QuantumMachine import QuantumMachine

from qcore.instruments import QM
from qcore.helpers.logger import logger
from qcore.modes.mode import Mode

# from qualang_tools.octave_tools.calibration_result_plotter import CalibrationResultPlotter

class OctaveMixerTuner:


    def __init__(self, modes_to_tune: tuple[Mode], qcore_qm : QM):
        self.modes_to_tune = modes_to_tune
        self.qcore_qm = qcore_qm

    def tune_mixers(self):
        qm = self.qcore_qm._qm
        for mode in self.modes_to_tune:
            logger.info(f"Tuning {mode.name} mixers ...")
            qm.calibrate_element(mode.name)

