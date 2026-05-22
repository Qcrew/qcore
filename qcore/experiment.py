""" """

from datetime import datetime

from contextlib import ExitStack
from pathlib import Path
import time

import qm.qua as qua
from qm.qua._dsl import _Variable, _ResultSource
from qm import generate_qua_script, Program
from qcore.instruments.instrument import Instrument
from qcore.instruments import QM
from qcore.helpers.datasaver import Datasaver
from qcore.helpers.logger import logger
from qcore.helpers.plotter import Plotter
from qcore.helpers.stage import Stage
from qcore.libs.qua_macros import QuaVariable
from qcore.modes.mode import Mode
from qcore.pulses.pulse import Pulse
from qcore.resource import Resource
from qcore.variables.datasets import Dataset
from qcore.variables.sweeps import Sweep


class ExperimentInitializationError(Exception):
    """ """


class SweepValidationError(Exception):
    """ """


class DatasetInitializationError(Exception):
    """ """


class ExperimentManager:
    """handle Experiment setup tasks related to resources, sweeps, datasets"""

    def get_resources(self, folder) -> dict[str, Instrument]:
        """get all available resources from remote stage and local config"""
        with Stage(remote=True) as stage:
            instruments = {rsc.name: rsc for rsc in stage.get(*stage.resources)}

        modes_config = folder / "config/modes.yml"
        if not modes_config.exists():
            message = f"A file named 'modes.yml' must exist at path '{modes_config}'."
            raise ExperimentInitializationError(message)

        with Stage(modes_config) as stage:
            rscs = stage.get(*stage.resources)
            modes = {m.name: m for m in rscs if isinstance(m, Mode)}
        pulses = {p.name: p for m in modes.values() for p in m.operations.values()}
        return instruments, modes, pulses

    def select_resources(self, resources, map, cls) -> dict[str, Resource]:
        """ """
        selected_resources = {}
        for key, value in map.items():
            try:
                resource = resources[value]
            except KeyError:
                message = f"Resource named '{value}' does not exist on the stage."
                logger.error(message)
                raise ExperimentInitializationError(message)
            else:
                if not isinstance(resource, cls):
                    message = (
                        f"Expect Resource named '{value}' to be of {cls}, got "
                        f"{resource} of {type(resource)}."
                    )
                    logger.error(message)
                    raise ExperimentInitializationError(message)
                selected_resources[key] = resource
        return selected_resources

    def select_modes(self, all_modes, mode_map) -> dict[str, Mode]:
        """ """
        return self.select_resources(all_modes, mode_map, Mode)

    def select_pulses(self, all_pulses, pulse_map) -> dict[str, Pulse]:
        """ """
        return self.select_resources(all_pulses, pulse_map, Pulse)

    def validate_sweeps(
        self, sweeps: list[Sweep], primary_sweeps: list[str], **kwargs
    ) -> None:
        """ """
        sweep_dict = {sweep.name: sweep for sweep in sweeps}

        # check if maximum number of Sweeps is exceeded
        if len(sweep_dict) > Experiment.MAX_SWEEPS:
            message = (
                f"Maximum sweeps for an experiment run exceeded, we only allow "
                f"upto {Experiment.MAX_SWEEPS}D Sweeps."
            )
            logger.error(message)
            raise SweepValidationError(message)

        # check that all values are Sweep objects
        for sweep in sweep_dict.values():
            if not isinstance(sweep, Sweep):
                msg = f"Expect object of '{Sweep}', got '{sweep}' of '{type(sweep)}'."
                logger.error(msg)
                raise SweepValidationError(msg)

        # check that required primary sweep(s) are specified
        if not set(primary_sweeps) <= set(sweep_dict.keys()):
            missing_sweeps = set(primary_sweeps) - set(sweep_dict.keys())
            message = f"All {primary_sweeps = } must be specified, {missing_sweeps = }."
            logger.error(message)
            raise SweepValidationError(message)

        # check that there are no duplicate sweep names
        if len(sweeps) != len(sweep_dict):
            all_names = [sweep.name for sweep in sweeps]
            duplicate_sweeps = set(i for i in all_names if all_names.count(i) > 1)
            message = f"Found {duplicate_sweeps = }, all sweep names must be unique."
            logger.error(message)
            raise SweepValidationError(message)

        # sweep names cannot appear in any control parameters dict
        for key in kwargs.keys():
            if key in sweep_dict:
                message = (
                    f"The variable with name '{key}' cannot be both specified as "
                    f"a Sweep and as an experimental control parameter."
                )
                logger.error(message)
                raise SweepValidationError(message)

    def init_sweeps(self, sweeps: list[Sweep]):
        """ """
        qcore_sweeps, qua_sweeps = [], []
        qcore_msg = "Among all sweeps, only 1 outermost Qcore Sweep can be specified."
        qua_msg = (
            "Among Qua Sweeps, atleast 1 outermost averaging Qua Sweep named 'N' must "
            "be specified."
        )

        for idx, sweep in enumerate(sweeps):
            if sweep.is_qua_sweep:
                qua_sweeps.append(sweep)
            else:
                if idx != 0:
                    logger.error(qcore_msg)
                    raise SweepValidationError(qcore_msg)
                qcore_sweeps.append(sweep)

            sweep.initialize()

        if len(qcore_sweeps) > 1:
            logger.error(qcore_msg)
            raise SweepValidationError(qcore_msg)

        try:
            outermost_qua_sweep = qua_sweeps[0]
        except IndexError:
            logger.error(qua_msg)
            raise SweepValidationError(qua_msg)
        else:
            if outermost_qua_sweep.name != "N":
                logger.error(qua_msg)
                raise SweepValidationError(qua_msg)

    def validate_datasets(
        self,
        datasets: list[Dataset],
        primary_datasets: list[str],
        sweep_dict: dict[str, Sweep],
    ) -> None:
        """ """
        dset_dict = {dataset.name: dataset for dataset in datasets}

        # check that all values are Dataset objects
        for dset in dset_dict.values():
            if not isinstance(dset, Dataset):
                msg = f"Expect object of '{Dataset}', got '{dset}' of '{type(dset)}'."
                logger.error(msg)
                raise DatasetInitializationError(msg)

        # all primary datasets must be specified
        if not set(primary_datasets) <= set(dset_dict.keys()):
            missing_datasets = set(primary_datasets) - set(dset_dict.keys())
            msg = f"All {primary_datasets = } must be specified, {missing_datasets = }."
            logger.error(msg)
            raise DatasetInitializationError(msg)

        # check that there are no duplicate dataset names
        if len(datasets) != len(dset_dict):
            all_names = [dset.name for dset in datasets]
            duplicate_datasets = set(i for i in all_names if all_names.count(i) > 1)
            msg = f"Found {duplicate_datasets = }, all dataset names must be unique."
            logger.error(msg)
            raise DatasetInitializationError(msg)

        # check that Datasets and Sweeps do not share any common names
        common_names = set(dset_dict.keys()) & set(sweep_dict.keys())
        if common_names:
            message = f"Datasets and Sweeps can't share names, found {common_names = }."
            logger.error(message)
            raise DatasetInitializationError(message)

    def init_datasets(
        self,
        datasets: dict[str, Dataset],
        primary_datasets: list[str],
        sweep_dict: dict[str, Sweep],
    ) -> None:
        """ """
        # prepare Dataset axes
        for dset in datasets.values():
            if dset.name in primary_datasets:
                if dset.stream is False:  # do not change stream value for ADC datasets
                    dset.stream = True  # by convention, only raw datasets are streamed
            else:
                if not dset.inputs:
                    dset.inputs = primary_datasets
                else:
                    for input_dset in dset.inputs:
                        if input_dset not in datasets and input_dset not in sweep_dict:
                            message = (
                                f"Derived dataset '{dset.name}' has an "
                                f"unrecognized input dataset named '{input_dset}'."
                            )
                            logger.error(message)
                            raise DatasetInitializationError(message)

            if dset.axes is None:
                dset.initialize(axes=list(sweep_dict.values()))


class Experiment:
    """generic experiment class written for executing QUA sequences on the QM OPX"""

    # datasets with these names will be streamed by the OPX
    primary_datasets: list = []  # to be specified by child classes

    # these name(s) must be specified as Sweep objects in the 'sweeps' list
    primary_sweeps: list = []  # to be specified by child classes

    MAX_SWEEPS: int = 6  # maximum number of Sweeps allowed per experiment run
    DATAFILE_SUFFIX: str = ".hdf5"

    def __init__(
        self,
        folder: Path,
        modes: dict[str, str],
        pulses: dict[str, str],
        sweeps: list[Sweep],
        datasets: list[Dataset],
        current_value: int = None,
        fetch_interval: int = 1,
        **kwargs,
    ) -> None:
        """ """
        if current_value is None:
            self.name = self.__class__.__name__
        else:
            self.name = "_".join([self.__class__.__name__, str(current_value*1e3), "mA"])
            # self.name = self.__class__.__name__ + "/" + str(current_value*1e3)

        self._folder = Path(folder)
        self._filepath = None  # will be set on run() with call to _get_filepath()

        self._manager = ExperimentManager()  # to handle Experiment setup tasks
        instruments, all_modes, all_pulses = self._manager.get_resources(self._folder)

        self._resources = {**instruments, **all_modes, **all_pulses}
        self.instruments = instruments
        self.modes = self._manager.select_modes(all_modes, modes)
        self.pulses = self._manager.select_pulses(all_pulses, pulses)
        self._configure_resources()

        # process Sweeps specified by the user
        primary_sweeps, primary_datasets = self.primary_sweeps, self.primary_datasets
        self._manager.validate_sweeps(sweeps, primary_sweeps, **kwargs)
        self.sweeps: dict[str, Sweep] = {swp.name: swp for swp in sweeps}
        self._manager.init_sweeps(sweeps)

        # process Datasets specified by the user
        self._manager.validate_datasets(datasets, primary_datasets, self.sweeps)
        self.datasets: dict[str, Dataset] = {dset.name: dset for dset in datasets}

        self.fetch_interval = fetch_interval

        # container for the various types of QuaVariables involved in this experiment
        self._qua_variables: dict[str, QuaVariable] = {}  # for all QuaVariables
        self._qua_sweeps: dict[str, Sweep] = {}
        self._qua_datasets = {}
        for k, v in kwargs.items():
            if isinstance(v, QuaVariable):
                v.tag = k
                self._qua_variables[k] = v
            else:
                setattr(self, k, v)  # set additional "kwargs" as Experiment attributes

        self.repetitions = 1
        for sweep in self.sweeps.values():
            if sweep.is_qua_sweep:
                self._qua_variables[sweep.name] = sweep
                self._qua_sweeps[sweep.name] = sweep

                if sweep.name == "N":
                    self.repetitions = sweep.length
        self._manager.init_datasets(self.datasets, primary_datasets, self._qua_sweeps)
        for dataset in self.datasets.values():
            if dataset.stream:  # is qua dataset
                self._qua_variables[dataset.name] = dataset
                self._qua_datasets[dataset.name] = dataset

        # initialize experiment attributes that will be set on run()
        self._qm = None

    def sequence(self):
        raise NotImplementedError("Subclass(es) to implement sequence()")

    def _configure_resources(self) -> None:
        """
        # make Mode and Pulse objects Experiment attributes for easy access
        # for each Mode, select only the subset of Pulses required for this Experiment
        """
        for mode_name, mode in self.modes.items():
            if not hasattr(self, mode_name):
                setattr(self, mode_name, mode)
                logger.info(f"Set '{self.name}' attribute '{mode_name}'.")

            all_op_names = [p.name for p in mode.operations.values()]
            selected_operations = {}
            for pulse_name, pulse in self.pulses.items():
                if pulse.name in all_op_names:
                    selected_operations[pulse_name] = pulse
                    if not hasattr(self, pulse_name):
                        setattr(self, pulse_name, pulse)
                        logger.info(f"Set '{self.name}' attribute '{pulse_name}'.")
            mode.operations = selected_operations

    def run(self,simulate=False, filename_suffix=None):
        """ """
        outermost_sweep = list(self.sweeps.values())[0]
        try:
            if not outermost_sweep.is_qua_sweep:
                self._run_with_qcore_sweep(outermost_sweep)
            else:
                self._get_filepath(filename_suffix)
            if simulate:
                self._run_qua_sweeps_simulate()
            else:
                self._run_qua_sweeps()
    
        except KeyboardInterrupt:
            msg = f"Experiment '{self.name}' interrupted, closing QM now..."
            logger.info(msg)
            self._qm.disconnect()

    def simulate(self):
        self._qm: QM = self._get_qm()
        qua_program = self._build_qua_program()
        self._qm.simulate(qua_program, self.repetitions)

    def _run_with_qcore_sweep(self, qcore_sweep: Sweep):
        """ """
        name, target, points = qcore_sweep.name, qcore_sweep.target, qcore_sweep.data

        if isinstance(target, str):  # target is a resource
            try:
                target = self._resources[target]
            except KeyError:
                message = f"Target Resource named '{target}' not found on stage."
                logger.error(message)
                raise SweepValidationError(message)
            else:
                if isinstance(target, Mode):
                    self.modes[target.name] = target
                elif isinstance(target, Pulse):
                    self.pulses[target.name] = target
        elif target is not self:
            message = f"Invalid sweep {target = } of type {type(target)}."
            logger.error(message)
            raise SweepValidationError(message)

        if not hasattr(target, name):
            message = f"Target '{target}' doesn't have attribute '{name}' for sweeping."
            logger.error(message)
            raise SweepValidationError(message)

        if qcore_sweep.dtype is str:
            has_resource_sweep_points = True
            try:
                points = [self._resources[point] for point in points]
            except KeyError:
                message = (
                    f"All strings specified as Qcore Sweep '{name}' points must be "
                    f"names of resources on the stage."
                )
                logger.error(message)
                raise SweepValidationError(message)
            else:
                for point in points:
                    if isinstance(point, Mode):
                        self.modes[point.name] = point
                    elif isinstance(point, Pulse):
                        self.pulses[point.name] = point

        self._configure_resources()

        for point in points:
            setattr(target, name, point)
            suffix = point.name if has_resource_sweep_points else str(point)
            tag = f"_{target.name}_{suffix}"
            filepath = self._get_filepath()
            self._filepath = filepath.parent / (filepath.stem + tag + filepath.suffix)
            self._run_qua_sweeps(point, exit_plotter=True)
            time.sleep(self.fetch_interval)

    def _run_qua_sweeps(self, qcore_sweep_point=None, exit_plotter=False):
        """ """
        self._qm: QM = self._get_qm()
        qua_program = self._build_qua_program()

        with open("qua_prog.py", "w+") as f:
            f.write(generate_qua_script(qua_program, self._qm.get_config()))
        self._qm.execute(qua_program, self.repetitions)

        time.sleep(self.fetch_interval)

        dsets_to_save = {k: dset for k, dset in self.datasets.items() if dset.save}
        sweeps_to_save = {k: swp for k, swp in self._qua_sweeps.items() if swp.save}

        datasaver = Datasaver(self._filepath, *self.datasets.values())

        to_plot = [dset for dset in self.datasets.values() if dset.plot]
        if len(to_plot) > 0:
            plotter = Plotter(self.fetch_interval, self.name, self._filepath, *to_plot)
        else:
            plotter = None

        with datasaver:
            # datasaver.save_metadata(self.metadata)
            while self._qm.is_processing():
                if plotter and plotter.stop_expt:
                    break

                # fetch latest batch of partial data along with data counts
                data, prev_count, incoming_count = self._qm.fetch()
                plot_msg = f": {incoming_count} / {self.repetitions} data batches"
                if data:  # to prevent update when empty data dict is fetched
                    # update sweep data and save to datafile
                    for name, sweep in sweeps_to_save.items():
                        if sweep.is_qua_sweep:
                            sweep.update(data[name])
                            datasaver.save_data(sweep)

                    # update primary datasets first
                    for name, dset in self.datasets.items():
                        if name in self.primary_datasets:
                            rawdata = (data[name], data[f"{name}_avg"])
                            dset.update(rawdata, prev_count, incoming_count)

                    # update derived datasets
                    for name, dset in self.datasets.items():
                        if dset.inputs:  # is derived dataset with datafn and inputs
                            dsets = []
                            for i in dset.inputs:
                                if i in self.datasets:
                                    dsets.append(self.datasets[i])
                                elif i in self.sweeps:
                                    dsets.append(self.sweeps[i])
                            dset.update(dsets, prev_count, incoming_count)
                            data[name] = dset.data

                    # process additional user-defined datasets in subclasses
                    self.process_data(
                        data,
                        prev_count,
                        incoming_count,
                        qcore_sweep_point,
                    )

                    # save datasets and sweeps (after updating) to datafile
                    for name, dataset in dsets_to_save.items():
                        datasaver.save_data(dataset)

                if plotter:
                    plotter.plot(message=plot_msg)  # update live plot

                time.sleep(self.fetch_interval)

            self._qm.disconnect()
            logger.info(f"{self.name} experiment has stopped running!")

            # plot final data batch and stop plotting loop
            if plotter:
                if exit_plotter:
                    plotter.plot(message=f"{plot_msg} [DONE]", stop=True, exit=True)
                else:
                    plotter.plot(message=f"{plot_msg} [DONE]", stop=True)

    def _run_qua_sweeps_simulate(self, qcore_sweep_point=None, exit_plotter=False):
        """ """
        self._qm: QM = self._get_qm()
        qua_program = self._build_qua_program()
        #####
        from qm import SimulationConfig
        import matplotlib.pyplot as plt
        from qm import generate_qua_script
        import os
        import numpy as np
        import plotly.graph_objects as go

        import json
        # sourceFile = open('debug.py', 'w')
        # print(generate_qua_script(qua_program, self._qm.get_config()))
        # sourceFile.close()

        # # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=40000//4)  # In clock cycles = 4ns
        # Simulate blocks python until the simulation is done
        job = self._qm._qmm.simulate(self._qm.get_config(), qua_program, simulation_config)
        
        with open("qua_prog.py", "w+") as f:
            f.write(generate_qua_script(qua_program, self._qm.get_config()))
        self._qm.execute(qua_program, self.repetitions)
        # Define the directory where you want to save the file
        # save_dir = r"C:\Users\qcrew\Documents\simulated_data"
        # os.makedirs(save_dir, exist_ok=True)
        # file_path = os.path.join(save_dir, "simulated_samples.json")

        data = job.get_simulated_samples()  # Assuming this returns a dictionary or list
        data.con1.plot()
        # Plot the simulated samples

        fig = go.Figure()

        # List of (key, label) pairs
        analog_keys = [
            ("3-1-1", "rrA"),
            ("3-2-1", "empty"),
            ("3-3-1", "qC"),
            ("3-3-2", "drive"),
            ("3-4-1", "cavB"),
            ("3-5-1", "cavA"),
            ("3-6-1", "qA"),
            ("3-7-1", "qB"),
            ("3-7-2", "probe"),
            ("3-8-1", "rrB"),
            ("1-8-1", "rrC"),
        ]
        offset = 0
        for key, label in analog_keys:
            try:
                dataset = data.con1.analog[key] #data['con1']['analog'][key]
                y = np.real(np.array(dataset)) + offset
                offset += 0.1 # visual separation between traces
                if y.size == 0:
                    print(f"⚠️ {label} ({key}) is empty, skipping.")
                    continue
                
                fig.add_trace(go.Scatter(y=y, mode='lines', name=label))

            except KeyError:
                print(f"⚠️ Key '{key}' not found in data['con1']['analog'], skipping.")
            except Exception as e:
                print(f"⚠️ Error loading '{key}' ({label}): {e}")

        digital_keys = [
            ("1-1", "Digital marker 1"),
            ("1-2", "Digital marker 2"),
            ("1-3", "Digital marker 3"),
            ("1-4", "Digital marker 4"),
            ("1-5", "Digital marker 5"),
            ("1-6", "Digital marker 6"),
            ("1-7", "Digital marker 7"),
            ("1-8", "Digital marker 8"),
        ]

        for key, label in digital_keys:
            try:
                dataset = data.con1.digital[key] #data['con1']['analog'][key]
                y = np.real(np.array(dataset)) + offset
                offset += 0.1 # visual separation between traces
                if y.size == 0:
                    print(f"⚠️ {label} ({key}) is empty, skipping.")
                    continue
                
                fig.add_trace(go.Scatter(y=y, mode='lines', name=label))

            except KeyError:
                print(f"⚠️ Key '{key}' not found in data['con1']['digital'], skipping.")
            except Exception as e:
                print(f"⚠️ Error loading '{key}' ({label}): {e}")

        # #     # Customize layout
        fig.update_layout(title="QM Simulator",
                  xaxis_title="time(ns)",
                  yaxis_title="a.u.",
                  legend=dict(x=0, y=1))
        fig.show()


    def process_data(self, data, prev_count, incoming_count, qcore_sweep_point):
        """Subclass(es) to implement process_data()"""
        pass

    def _build_qua_program(self) -> Program:
        """ """
        # enter QUA program scope
        with qua.program() as qua_program:
            # declare QUA variables and streams
            # set those as self attributes for easy access
            for name, var in self._qua_variables.items():
                qua_variable = var.declare_variable()
                qua_stream = var.declare_stream()
                if var.is_adc_trace:
                    setattr(self, name, qua_stream)
                else:
                    setattr(self, name, qua_variable)
                logger.info(f"Set QUA variable attribute {name} for {self.name}.")

            # generate and enter QUA loop contexts programmatically
            with ExitStack() as stack:
                for name, sweep in self._qua_sweeps.items():
                    logger.debug(f"Expect {sweep.length} '{name}' sweep points.")
                    fn, *args = sweep.generate_loop()
                    stack.enter_context(fn(*args))
                    sweep.save_to_stream()
                self.sequence()
                for dataset in self._qua_datasets.values():
                    dataset.save_to_stream()

            with qua.stream_processing():
                for idx, (sweep) in enumerate(self._qua_sweeps.values()):
                    if idx != 0:  # we don't save repetitions at all
                        sweep.process_stream()
                for dataset in self._qua_datasets.values():
                    dataset.process_stream()
        return qua_program

    def _get_qm(self):
        """pre-requisite: remote stage must already be setup and serving instruments"""
        mode_lo_map = {}
        for name, mode in self.modes.items():
            lo_name = mode.lo_name
            if lo_name is not None:
                try:
                    mode_lo_map[mode] = self.instruments[lo_name]
                except KeyError:
                    message = f"'{lo_name = }' for Mode '{name}' not found on stage."
                    logger.error(message)
                    raise ExperimentInitializationError(message)

        return QM(
            modes=mode_lo_map.keys(),
            oscillators=mode_lo_map.values(),
            opx=self.instruments.get("opx_plus", self.instruments.get("opx1000")),
            config_path=f"{self._folder}/config"
        )

    def _get_filepath(self, filename_suffix = None) -> Path:
        """ """
        if self._filepath is None:
            date, time = datetime.now().strftime("%Y-%m-%d %H-%M-%S").split()
            datafolder = self._folder / "data"
            datafolder.mkdir(exist_ok=True)
            folderpath = datafolder / date
            if filename_suffix:
                filename = f"{time}_{self.name}_{filename_suffix}{Experiment.DATAFILE_SUFFIX}"
            else:
                filename = f"{time}_{self.name}{Experiment.DATAFILE_SUFFIX}"
            self._filepath = folderpath / filename
            logger.debug(f"Generated filepath {self._filepath} for '{self.name}'")
        return self._filepath
    
    @property
    def metadata(self):
        """ """
        inst_mdata = {k: i.snapshot() for k, i in self.instruments.items()}
        mode_mdata = {k: m.snapshot(flatten=True) for k, m in self.modes.items()}

        xcls = (_Variable, Resource, _ResultSource)  # excluded classes
        xkeys = ("instruments", "modes", "pulses", "sweeps", "datasets")
        snapshot = {}
        for k, v in self.__dict__.items():
            if not isinstance(v, xcls) and not k.startswith("_") and not k in xkeys:
                snapshot[k] = v

        return {"instruments": inst_mdata, "modes": mode_mdata, None: snapshot}
