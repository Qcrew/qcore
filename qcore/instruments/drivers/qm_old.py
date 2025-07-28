""" """

import json
import numpy as np

from qm.QuantumMachine import QuantumMachine
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.QmJob import QmJob
from qm.qua._dsl import _ProgramScope
from qm.octave import QmOctaveConfig
from qcore.instruments.drivers.qm_octave_setter import OctaveUnit, octave_declaration
from qcore.helpers.logger import logger
from qcore.instruments.instrument import Instrument, ConnectionError
from qcore.instruments.drivers.qm_config_builder import QMConfigBuilder, QMConfig
from qcore.instruments.drivers.qm_result_fetcher import QMResultFetcher
from qcore.instruments.drivers.vaunix_lms import LMS
from qcore.instruments.drivers.qm_octave_dummy import Octave
from qcore.instruments.drivers.qm_opx_plus_dummy import OPXPlus
from qcore.instruments.drivers.qm_opx_one import OPXOne
from qcore.instruments.drivers.qm_opx import OPX1000
from qcore.modes.mode import Mode


class QM(Instrument):
    """By convention, we ensure only one QM is open at a given time"""

    QMM_PORT: int = 9510  # this works but 80 does not

    def __init__(
        self,
        modes: tuple[Mode] = None,
        oscillators: tuple[LMS] = None,
        opx_plus: OPXPlus = None,
        opx_one: OPXOne = None,
        opx: OPX1000 = None
    ) -> None:
        """ """
        self._status: bool = None

        self._qmm: QuantumMachinesManager = None
        self._qm: QuantumMachine = None
        self._config: QMConfig = None
        self._qcb: QMConfigBuilder = QMConfigBuilder()

        self._modes: tuple[Mode] = modes
        self._oscillators: tuple[LMS] = oscillators

        self._job: QmJob = None
        self._qrf: QMResultFetcher = None

        self._opx_plus = opx_plus
        self._opx_one = opx_one
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
            if self.requires_octave():
                if self.uses_opx_plus():
                    self._connect_to_opx_plus_and_octave()
                else:
                    self._connect_to_opx_one_and_octave()
            else:
                if self.uses_opx_plus():
                    self._connect_to_opx_plus()
                else:
                    self._connect_to_opx_one()
        except Exception as err:
            raise ConnectionError(f"Failed to connect QM. Details: {err}.") from None
        else:
            self._status = True
            if self._modes is not None and self._oscillators is not None:
                self.open(self._modes, self._oscillators, self._opx_plus)

    def _connect_to_opx_plus_and_octave(self):
        octave_config = self._make_octave_config()
        self._connect_to_opx_plus(octave_config)
    
    def _connect_to_opx_one_and_octave(self):
        octave_config = self._make_octave_config()
        self._connect_to_opx_one(octave_config)

    def _connect_to_opx_plus(self, octave_config=None):
        self._qmm = QuantumMachinesManager(
            host=self._opx_plus.id,
            port=None,
            cluster_name=self._opx_plus.cluster_name,
            octave=octave_config,
        )

    def _connect_to_opx_one(self, octave_config=None):
        if octave_config is None:
            self._qmm = QuantumMachinesManager(host=self._opx_one.id,port=QM.QMM_PORT)
        else:
            self._qmm = QuantumMachinesManager(host=self._opx_one.id,port=QM.QMM_PORT, octave=octave_config)

    def _make_octave_config(self) -> QmOctaveConfig:
        octaves = []
        for name, octave in self.get_octaves().items():
            octaves.append(OctaveUnit(name, octave.id, port=octave.port, con="con1"))

        octave_calibration_db_paths = set(
            [octave.calibration_db_path for octave in self.get_octaves().values()]
        )
        assert (
            len(octave_calibration_db_paths) == 1
        ), "Currently support only one calibration_db"

        octave_config = octave_declaration(
            octaves, calibration_db_path=octave_calibration_db_paths.pop()
        )

        return octave_config

    def open(
        self, modes: tuple[Mode], oscillators: tuple[LMS], opx_plus: OPXPlus = None
    ) -> QuantumMachine:
        """ """
        self._config = self._qcb.build_config(modes, oscillators, opx_plus)
        self._qm = self._qmm.open_qm(self._config, close_other_machines=True)
        return self._qm

    def get_config(self) -> dict:
        """ """
        return self._qm.get_config()

    def disconnect(self) -> None:
        """ """
        if self._qm is not None:
            self._qm.close()
        self._qmm.close()
        self._qmm = None
        self._status = False

    @property
    def status(self) -> bool:
        """ """
        return self._status

    def execute(self, qua_program: _ProgramScope, total_count=None):
        """ """
        if self._config is None or self._qm is None:
            logger.warning("Can't execute program, QM hasn't been opened with a config")
        else:
            # TODO error handling, if exception, set status = False, else set True
            self._job = self._qm.execute(qua_program)
            self._qrf = QMResultFetcher(self._job.result_handles, total_count)
            return self._job

    def simulate(self, qua_program: _ProgramScope, total_count=None):
        """ """
        if self._config is None or self._qm is None:
            logger.warning("Can't execute program, QM hasn't been opened with a config")
        else:
            # TODO error handling, if exception, set status = False, else set True
            self._job = self._qm.simulate(qua_program)
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
            if "octave" in oscillator.name:
                octaves[oscillator.name] = oscillator
        return octaves

    def uses_opx_plus(self) -> bool:
        return self._opx_plus is not None
