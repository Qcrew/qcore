""" """
import json
import numpy as np
from qm import SimulationConfig

from qm.QuantumMachine import QuantumMachine
from qm import QuantumMachinesManager, Program
from qm import QmJob
from qm.api.v2.qmm_api import ControllerOPX1000

from qm.octave import QmOctaveConfig
from qcore.instruments.drivers.qm_octave_setter import OctaveUnit, octave_declaration
from qcore.helpers.logger import logger
from qcore.instruments.instrument import Instrument, ConnectionError
from qcore.instruments.drivers.qm_config_builder import QMConfigBuilder, QMConfig
from qcore.instruments.drivers.qm_result_fetcher import QMResultFetcher
from qcore.instruments.drivers.vaunix_lms import LMS
from qcore.instruments.drivers.qm_octave_dummy import Octave
from qcore.instruments.drivers.qm_opx import OPXPlus, OPX1000, OPX
from qcore.modes.mode import Mode


class QM(Instrument):
    """By convention, we ensure only one QM is open at a given time"""

    QMM_PORT: int = 9510  # this works but 80 does not

    def __init__(
        self,
        modes: tuple[Mode] = None,
        oscillators: tuple[LMS] = None,
        opx: OPX = None,
        config_path: str = None
    ) -> None:
        """ """
        self._status: bool = None

        self._qmm: QuantumMachinesManager = None
        self._qm: QuantumMachine = None
        self._config_path = config_path
        self._config: QMConfig = None

        self._qcb: QMConfigBuilder = QMConfigBuilder()

        self._modes: tuple[Mode] = modes
        self._oscillators: tuple[LMS] = oscillators

        self._job: QmJob = None
        self._qrf: QMResultFetcher = None

        self._opx = opx

        super().__init__(id=None, name="QM")

    def __repr__(self) -> str:
        """ """
        return self.__class__.__name__

    def connect(self) -> None:
        """ """
        if self._qmm is not None:
            self.disconnect()
        try:
    
            if self.uses_opx_plus() or self.uses_opx1000():
                self._connect_to_opx()
            else:
                self._connect_to_opx_one()

        except Exception as err:
            raise ConnectionError(f"Failed to connect QM. Details: {err}.") from None

        else:
            self._status = True
            if self._modes is not None and self._oscillators is not None:
                self.open(self._modes, self._oscillators, self._opx)

    def _connect_to_opx(self):
        self._qmm = QuantumMachinesManager(
            host=self._opx.id,
            port=None,
            cluster_name=self._opx.cluster_name,
            octave_calibration_db_path=self._config_path,
        )

    def _connect_to_opx_one(self):
        self._qmm = QuantumMachinesManager(port=QM.QMM_PORT)


    def open(
        self, modes: tuple[Mode], oscillators: tuple[LMS], opx: OPX = None
    ) -> QuantumMachine:
        """ """
        controllers_info = self.get_controllers_info()
        self._config = self._qcb.build_config(modes, oscillators, opx, controllers_info)
        with open("config.json", "w+") as f:
            json.dump(self._config, f, indent=4)
        self._qm = self._qmm.open_qm(self._config, close_other_machines=True)
        return self._qm


    def get_controllers_info(self):
        controllers = self._qmm.get_controllers()
        controllers_info = {}
        for controller in controllers:
            if isinstance(controller, ControllerOPX1000):
                controllers_info[controller.name] = controller.fems

        return controllers_info

    def get_config(self) -> dict:
        """ """
        return self._qm.get_config()

    def disconnect(self) -> None:
        """ """
        if self._qm is not None:
            self._qm.close()
        # self._qmm.close()
        self._qmm = None
        self._status = False

    @property
    def status(self) -> bool:
        """ """
        return self._status

    def simulate(self, qua_program: Program, total_count=None):
        """ """
        if self._config is None or self._qm is None:
            logger.warning("Can't execute program, QM hasn't been opened with a config")
        else:
            simulation_config = SimulationConfig(duration=10000)  # In clock cycles = 4ns
            job = self._qm.simulate(qua_program, simulation_config)
            samples = job.get_simulated_samples()
            waveform_report = job.get_simulated_waveform_report()
            waveform_report.create_plot(samples, plot=True, save_path="./")
            return self._job

    def execute(self, qua_program: Program, total_count=None):
        """ """
        if self._config is None or self._qm is None:
            logger.warning("Can't execute program, QM hasn't been opened with a config")
        else:
            # TODO error handling, if exception, set status = False, else set True
            self._job = self._qm.execute(qua_program)
            self._qrf = QMResultFetcher(self._job.result_handles, total_count)
            return self._job

    def is_processing(self) -> bool:
        """ """
        return not self._qrf.is_done_fetching

    def fetch(self) -> tuple[dict[str, np.ndarray], int, int]:
        """ """
        return (self._qrf.fetch(), *self._qrf.counts)

    def set_output_dc_offset_by_element(self, element: str, input: str, offset: float):
        """ """
        self._qm.set_output_dc_offset_by_element(element, input, offset)

    def requires_octave(self) -> bool:
        octaves = self.get_octaves()
        return len(octaves) > 0

    def get_octaves(self) -> dict[str, Octave]:
        octaves = {}
        for oscillator in self._oscillators:
            if "oct" in oscillator.name:
                octaves[oscillator.name] = oscillator
        return octaves

    def uses_opx1000(self) -> bool:
        return self._opx.type == 'opx1000'

    def uses_opx_plus(self) -> bool:
        return self._opx.type == 'opx_plus'
