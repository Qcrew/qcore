""" """

from datetime import datetime
import itertools
import h5py

from pathlib import Path
import numpy as np

from qcore.helpers.logger import logger
from qcore.instruments.drivers.anritsu_ms46522b import MS46522B


class VNADataSaver:
    """ """

    def __init__(
        self,
        filepath: Path,
        datasets,
        datashape,
        datagroup: str = "data",
        datatype: str = "f8",
    ) -> None:
        """ """

        _data_filename = str(filepath)
        if len(_data_filename) > 240:
            _data_filename = _data_filename[:240] + "and_so_on..."
        self.datafile = h5py.File(_data_filename, "a")
        self.datagroup = datagroup
        self.datatype = datatype
        self.datasets = {}
        for datasetname in datasets:
            name = f"{datagroup}/{datasetname}"
            dataset = self.datafile.create_dataset(
                name=name, shape=datashape, dtype=datatype
            )
            self.datasets[datasetname] = dataset
        self.datafile.flush()

    def __enter__(self):
        """ """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ """
        self.datafile.flush()
        self.datafile.close()

    def save_metadata(self, metadatadict) -> None:
        """ """
        for key, value in metadatadict.items():
            try:
                self.datafile.attrs[key] = value
            except TypeError:
                pass
                # print(f"WARNING! Unable to save {key = }, {value = } as metadata")
        self.datafile.flush()

    def save_data(self, data, pos=None) -> None:
        """ """
        for name, datastream in data.items():
            if pos is None:  # create new dataset
                name, dtype = f"{self.datagroup}/{name}", self.datatype
                self.datafile.create_dataset(name=name, data=datastream, dtype=dtype)
            # insert into existing dataset at given pos
            else:
                dataset = self.datasets[name]
                if len(pos) == 1:
                    rep_count = (*pos,)
                    dataset[rep_count] = datastream
                elif len(pos) == 2:
                    rep_count, var_count = pos
                    dataset[rep_count, var_count] = datastream
                elif len(pos) == 3:
                    rep_count, var_count_1, var_count_2 = pos
                    dataset[rep_count, var_count_1, var_count_2] = datastream
                else:
                    raise RuntimeError(f"WE DO NOT DO {len(pos)}D SWEEPS HERE...")
        self.datafile.flush()


class VNAExperiment:
    """ """

    def __init__(
        self,
        folder: str,
        vna: MS46522B,
        repetitions: int,
        powers: tuple,
        attenuation: tuple,
        current_index: int,
    ) -> None:

        self.vna = vna
        self.repetitions = repetitions
        self.current_index = current_index
        self._folder = Path(folder)
        self._filepath = None
        self.port1_attenuation, self.port2_attenuation = attenuation

        is_valid = isinstance(powers, (tuple, list)) and len(powers) == 2
        if not is_valid:
            raise ValueError(f"Expect tuple of length 2, got {powers = }")

        # only frequency sweep, no power sweep specified
        is_fsweep = all(map(lambda x: isinstance(x, (float, int)), powers))
        # power sweep specified
        is_fpsweep = any(map(lambda x: isinstance(x, (tuple, list, set)), powers))
        if is_fsweep:
            self._run = self._run_fsweep
            self.powers = powers
            self.datashape = (self.repetitions, vna.sweep_points)
        elif is_fpsweep:
            self._run = self._run_fpsweep
            powerlist = []
            for powerspec in powers:
                if isinstance(powerspec, (float, int)):
                    powerlist.append((powerspec,))
                elif isinstance(powerspec, (list, tuple)) and len(powerspec) == 3:
                    start, stop, step = powerspec
                    powerlist.append(np.arange(start, stop + step / 2, step))
                elif isinstance(powerspec, set):
                    powerlist.append(sorted(powerspec))
            self.powers = list(itertools.product(*powerlist))
            logger.info(f"Found {len(self.powers)} input power combinations specified")
            self.datashape = (self.repetitions, len(self.powers), vna.sweep_points)
        else:
            raise ValueError(f"Invalid specification of {powers = }")

    def run(self, metadata) -> None:
        # save frequency data since its already available and is the same for all sweeps
        with VNADataSaver(
            self._get_filepath(), self.vna.datakeys, self.datashape
        ) as saver:
            saver.save_metadata(metadata)
            saver.save_data({"frequency": self.vna.frequencies})  # arg must be a dict
            self._run(saver)  # runs fsweep or fpsweep based on sweep initialization

    def _run_fsweep(self, saver) -> None:
        """ """
        self.vna.powers = self.powers  # set input power on the VNA
        for rep in range(self.repetitions):
            data = self.vna.sweep()
            saver.save_data(data, pos=(rep,))  # save to root group
            logger.info(f"Frequency sweep count = {rep+1} / {self.repetitions}")

    def _run_fpsweep(self, saver) -> None:
        """ """
        # save power data since its already available
        powers = np.array(tuple(self.powers)).T
        port1_powers = powers[0] - self.port1_attenuation
        port2_powers = powers[1] - self.port2_attenuation
        saver.save_data({"power": port1_powers, "power2": port2_powers})

        # for each power tuple in self.powers, do fsweep, for n reps
        for power_count, (p1, p2) in enumerate(self.powers):  # p1, p2 = port powers
            self.vna.powers = (p1, p2)  # set input power on the VNA
            logger.info(f"Set power = ({p1}, {p2})")
            for rep in range(self.repetitions):
                data = self.vna.sweep()
                saver.save_data(data, pos=(rep, power_count))
                logger.info(f"Frequency sweep repetition {rep+1} / {self.repetitions}")
            if self.vna.is_averaging:
                self.vna.reset_averaging_count()

    def _get_filepath(self) -> Path:
        """ """
        if self._filepath is None:
            date, time = datetime.now().strftime("%Y-%m-%d %H-%M-%S").split()
            datafolder = self._folder / "data" / date
            datafolder.mkdir(exist_ok=True, parents=True)
            folderpath = datafolder
            filename = f"{time}_VNA_sweep_{self.vna.fcenter}_{self.powers}pow_{self.repetitions}reps_{self.current_index}.hdf5"
            # filename = f"{time}_VNA_sweep_{self.vna.fcenter}_{self.repetitions}reps_{self.current_index}.hdf5"
            self._filepath = folderpath / filename
            logger.debug(f"Generated filepath {self._filepath}")
        return self._filepath


def create_attribute(input_data: tuple) -> list:
    powerlist = []
    for powerspec in input_data:
        if isinstance(powerspec, (float, int)):
            powerlist.append((powerspec,))
        elif isinstance(powerspec, (list, tuple)) and len(powerspec) == 3:
            start, stop, step = powerspec
            powerlist.append(np.arange(start, stop + step / 2, step))
        elif isinstance(powerspec, set):
            powerlist.append(sorted(powerspec))
    return list(itertools.product(*powerlist))


class VNAExperiment_with_yoko:
    """ """

    def __init__(
        self,
        folder: str,
        vna: MS46522B,
        yoko: None,
        repetitions: int,
        powers: tuple,
        currents: tuple,
        attenuation: tuple,
        current_step: float = 1e-4,
    ) -> None:

        self.vna = vna
        if yoko is None:
            from qcore.instruments.drivers.yokogawa_gs200 import GS200

            self.yoko = GS200
        else:
            self.yoko = yoko
        self.repetitions = repetitions
        self.step = current_step
        self._folder = Path(folder)
        self._filepath = None
        self.port1_attenuation, self.port2_attenuation = attenuation

        is_valid = isinstance(powers, (tuple, list)) and len(powers) == 2
        if not is_valid:
            raise ValueError(f"Expect tuple of length 2, got {powers = }")

        is_valid_for_currents = isinstance(currents, (tuple, list, set, np.ndarray))
        if not is_valid_for_currents:
            raise ValueError(f"Expect tuple of currents, got {currents = }")

        # only frequency sweep, no power sweep specified
        is_fsweep = all(map(lambda x: isinstance(x, (float, int)), powers))
        # power sweep specified
        is_fpsweep = any(map(lambda x: isinstance(x, (tuple, list, set)), powers))
        is_currentsweep = len(currents) > 1
        # is_current_sweep = isinstance(x, (tuple, list, set))
        if is_fsweep and (not is_currentsweep):
            self._run = self._run_fsweep
            self.powers = powers
            self.datashape = (self.repetitions, vna.sweep_points)
        elif is_fpsweep:
            self.powers = create_attribute(powers)
            if is_valid_for_currents:
                self._run = self._run_current_sweep
                self.currents = create_attribute(currents)[0]
                logger.info(
                    f"Found {len(self.currents)} currents combinations specified"
                )
                self.datashape = (
                    self.repetitions,
                    len(self.currents),
                    len(self.powers),
                    vna.sweep_points,
                )
            else:
                self._run = self._run_fpsweep
                self.datashape = (self.repetitions, len(self.powers), vna.sweep_points)
            logger.info(f"Found {len(self.powers)} input power combinations specified")
        elif is_currentsweep:
            self.powers = create_attribute(powers)
            self._run = self._run_current_sweep
            self.currents = create_attribute(currents)[0]
            logger.info(f"Found {len(self.currents)} currents combinations specified")
            self.datashape = (
                self.repetitions,
                len(self.currents),
                len(self.powers),
                vna.sweep_points,
            )
        else:
            raise ValueError(f"Invalid specification of {powers = } or {currents = }")

    def run(self, metadata) -> None:
        # save frequency data since its already available and is the same for all sweeps
        with VNADataSaver(
            self._get_filepath(), self.vna.datakeys, self.datashape
        ) as saver:
            saver.save_metadata(metadata)
            saver.save_data({"frequency": self.vna.frequencies})  # arg must be a dict
            self._run(saver)  # runs fsweep or fpsweep based on sweep initialization

    def _run_fsweep(self, saver) -> None:
        """ """
        self.vna.powers = self.powers  # set input power on the VNA
        for rep in range(self.repetitions):
            data = self.vna.sweep()
            saver.save_data(data, pos=(rep,))  # save to root group
            logger.info(f"Frequency sweep count = {rep+1} / {self.repetitions}")

    def _run_fpsweep(self, saver) -> None:
        """ """
        # save power data since its already available
        powers = np.array(tuple(self.powers)).T
        port1_powers = powers[0] - self.port1_attenuation
        port2_powers = powers[1] - self.port2_attenuation
        saver.save_data({"power": port1_powers, "power2": port2_powers})

        # for each power tuple in self.powers, do fsweep, for n reps
        for power_count, (p1, p2) in enumerate(self.powers):  # p1, p2 = port powers
            self.vna.powers = (p1, p2)  # set input power on the VNA
            logger.info(f"Set power = ({p1}, {p2})")
            for rep in range(self.repetitions):
                data = self.vna.sweep()
                saver.save_data(data, pos=(rep, power_count))
                logger.info(f"Frequency sweep repetition {rep+1} / {self.repetitions}")
            if self.vna.is_averaging:
                self.vna.reset_averaging_count()

    def _run_current_sweep(self, saver) -> None:
        """VNA sweep with simultaneous Yokogawa current adjustments"""
        # save power data since its already available
        powers = np.array(tuple(self.powers)).T
        currents = np.array(tuple(self.currents)).T
        port1_powers = powers[0] - self.port1_attenuation
        port2_powers = powers[1] - self.port2_attenuation
        saver.save_data(
            {"power1": port1_powers, "power2": port2_powers, "currents": currents}
        )

        # for each current in self.currents and each power tuple in self.powers, do fsweep, for n reps
        for current_count, current in enumerate(self.currents):
            self.yoko.ramp(stop=current, step=self.step)
            logger.info(f"Set current = {current*1e3} mA")
            for power_count, (p1, p2) in enumerate(self.powers):  # p1, p2 = port powers
                self.vna.powers = (p1, p2)  # set input power on the VNA
                logger.info(f"Set power = ({p1}, {p2}) dBm")
                for rep in range(self.repetitions):
                    data = self.vna.sweep()
                    saver.save_data(data, pos=(rep, current_count, power_count))
                    logger.info(
                        f"Frequency sweep repetition {rep+1} / {self.repetitions}"
                    )
                if self.vna.is_averaging:
                    self.vna.reset_averaging_count()

    def _get_filepath(self) -> Path:
        """ """
        if self._filepath is None:
            date, time = datetime.now().strftime("%Y-%m-%d %H-%M-%S").split()
            datafolder = self._folder / "data" / date
            datafolder.mkdir(exist_ok=True)
            folderpath = datafolder
            filename = f"{time}_VNA_sweep_{self.vna.fcenter}_{self.powers}pow_{self.repetitions}reps.hdf5"
            self._filepath = folderpath / filename
            logger.debug(f"Generated filepath {self._filepath}")
        return self._filepath
