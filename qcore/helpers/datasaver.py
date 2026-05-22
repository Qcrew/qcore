""" """

from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import Union

import h5py
import numpy as np

from qcore.variables.datasets import Dataset
from qcore.helpers.logger import logger
from qcore.variables.sweeps import Sweep


class DataSavingError(Exception):
    """ """


class Datasaver:
    """ """

    def __init__(self, path: Path, *datasets: Dataset) -> None:
        """
        path: full path str to the datafile (must end in .h5 or .hdf5). DataSaver is not responsible for setting datafile naming/saving convention, the caller is.
        *datasets: Dataset objects to be saved.
        """
        self._file = None  # internal reference to the hdf5 file

        self._dataspec: dict[str, Union[Dataset, Sweep]] = {}  # internal attr
        self._datalog = {}  # to track Dataset size during saving

        self._path = path
        self._path.parent.mkdir(exist_ok=True)  # avoid IOError due to missing directory

        self._create_datasets(*datasets)

        logger.debug(f"Initialized a Datasaver tagged to data file at {self._path}.")

    def _create_datasets(self, *datasets: Dataset) -> None:
        """ """
        # mode = "x" means create file, fail if exists
        with h5py.File(self._path, mode="x", track_order=True) as file:
            coordinates = self._find_coordinates(*datasets)  # find sweeps of indep vars
            for name, sweep in coordinates.items():  # create coordinate datasets first
                if sweep.save:
                    self._create_dataset(file, shape=sweep.shape, **sweep.metadata)
                    self._dataspec[name] = sweep

            for dataset in datasets:
                if dataset.save:
                    self._create_dataset(file, shape=dataset.shape, **dataset.metadata)
                    self._dataspec[dataset.name] = dataset
                    self._dimensionalize_dataset(file, dataset)

    def _find_coordinates(self, *datasets: Dataset) -> dict[str, Sweep]:
        """coordinate datasets hold the data of Sweeps"""
        coordinates = {}  # dict prevents duplication of Sweeps
        for dataset in datasets:
            for value in dataset.axes:
                if isinstance(value, Sweep):
                    coordinates[value.name] = value
        logger.debug(f"Found {len(coordinates)} coordinates in the dataspec.")
        return coordinates

    def _create_dataset(
        self,
        file: h5py.File,
        name: str,
        shape: tuple[int],
        dtype: str,
        chunks: Union[bool, tuple[int]] = True,
        **metadata,
    ) -> None:
        """wrapper for h5py method. default fillvalue decided by h5py. metadata kwargs will be saved as dataset attrs"""
        # by default, we create resizable datasets with shape = maxshape
        # we resize the dataset in __exit__() after all data is written to it
        dataset = file.create_dataset(
            name=name,
            shape=shape,
            maxshape=shape,
            chunks=chunks,
            dtype=dtype,
            track_order=True,
        )

        for key, value in metadata.items():
            value = self._parse_attribute(key, value)
            dataset.attrs[key] = value

        logger.debug(f"Created dataset '{name}' with {shape = } in {self._path.name}.")

    def _dimensionalize_dataset(self, file: h5py.File, dataset: Dataset) -> None:
        """internal method for attaching dimension scales to a single dataset"""
        h5dset = file[dataset.name]  # h5py Dataset is different from a qcore Dataset
        labels = [ax.name if isinstance(ax, Sweep) else None for ax in dataset.axes]
        for idx, label in enumerate(labels):
            if label is not None:
                h5dset.dims[idx].label = label  # make dimension label
                logger.debug(f"Set dataset '{dataset.name}' dim {idx} '{label = }'.")
                if label in file.keys():
                    coordinate = file[label]
                    coordinate.make_scale(label)
                    h5dset.dims[idx].attach_scale(coordinate)
                    logger.debug(f"Attached dataset '{dataset.name}' scale '{label}'.")

    def __enter__(self) -> Datasaver:
        """ """
        # 'r+' means read/write, file must exist
        self._file = h5py.File(self._path, mode="r+")
        logger.debug(f"Started DataSaver session tagged to '{self._file.filename}'.")

        # track the maximum value of the index the data is written to for each dimension
        # this will allow us to trim reziable datasets and mark uninitialized ones
        for name, dataset in self._dataspec.items():
            self._datalog[name] = [0] * len(dataset.shape)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """ """
        # trim datasets
        for name, dataset in self._dataspec.items():
            fin_shape = tuple(self._datalog[name])
            init_shape = dataset.shape

            if all(idx == 0 for idx in fin_shape):  # dataset has not been written into
                del self._file[name]  # delete dataset
                logger.debug(f"Deleted dataset '{name}' as it was not written into.")
            elif fin_shape != init_shape:  # dataset has been partially written into
                self._file[name].resize(fin_shape)  # trim dataset
                logger.debug(f"Resized dataset '{name}': {init_shape} -> {fin_shape}.")

        self._file.close()
        self._file = None

    def _validate_session(self) -> None:
        """check if hdf5 file is currently open (called when either save_data() or save_metadata() is called). enforces use of DataSaver context manager as the only means of writing to the data file."""
        if self._file is None:
            message = (
                f"The data file is not open. Please call data saving methods within a "
                f"DataSaver context manager and try again."
            )
            logger.error(message)
            raise DataSavingError(message)

    def save_data(self, dataset: Union[Dataset, Sweep]) -> None:
        """insert a batch of data to the dataset at specified index. please call this method within a datasaver context, if not it will throw error.

        dataset: Dataset the dataset as declared in the dataspec. raises an error if we encounter a dataset that has not been declared prior to saving.

        the dataset attributes 'data' and 'index' must be already configured prior to calling this method.

        dataset.data: np.ndarray incoming data to be written into dataset
        dataset.index = ... means that the incoming data is written to the entire dataset in one go i.e. we do dataset[...] = incoming_data. Use this when all the data to be saved is available in memory at the same time. this is the default option.
        dataset.index = tuple[slice | ... | int] means that you want to insert the incoming data to a specific location ("hyperslab") in the dataset. Use this while saving data that is being streamed in successive batches or in any other application that requires appending to existing dataset. we pass the index directly to h5py i.e. we do dataset[index] = incoming_data, so user must be familiar with h5py indexing convention to use this feature effectively. index must be a tuple (not list etc) to ensure proper saving behaviour.
        """
        self._validate_session()

        name, data = dataset.name, dataset.data
        # h5dset is a h5py Dataset, to distinguish it from our dataset
        h5dset = self._get_dataset(name)
        index = self._validate_index(dataset, h5dset)

        # EDIT 19/07/25 by JC: Temp fix to ADC bug <----------------------------------------------------------------------------------------------------
        # h5dset[index] = np.real(data)

        h5dset[index] = np.real(data)
        self._file.flush()

        shape = data.shape if isinstance(data, np.ndarray) else len(data)
        logger.debug(f"Wrote data with {shape = } to dataset '{name}' at '{index = }'")

        self._track_size(name, index)  # for trimming dataset if needed in __exit__()

    def _get_dataset(self, name: str) -> h5py.Dataset:
        """ """
        try:
            return self._file[name]
        except KeyError:
            message = f"Dataset '{name}' does not exist in {self._file.filename}."
            logger.error(message)
            raise DataSavingError(message) from None

    def _validate_index(self, dataset: Dataset, h5dset: h5py.Dataset) -> None:
        """ """
        index = dataset.index
        if index is ...:  # single ellipsis is a valid index
            return index

        # isinstance check is necessary to ensure stable datasaving
        if not isinstance(index, tuple):
            message = (
                f"Expect index of {tuple}, got '{index}' of '{type(index)}' "
                f"while writing to dataset '{dataset.name}'."
            )
            logger.error(message)
            raise DataSavingError(message)

        # to allow tracking of written data, convert any ... to slice(None, None)
        # such that the length of the index equals the no. of dataset dimensions
        new_index, ndims = [], h5dset.ndim
        for i in range(ndims):
            if index[i] is ...:
                new_index.extend([slice(None, None) for _ in range(ndims - i)])
                break
            else:
                new_index.append(index[i])
        return tuple(new_index)

    def _track_size(self, name: str, index: tuple[Union[int, slice]]) -> None:
        """ """
        if index is ...:  # we have written to the entire dataset
            self._datalog[name] = self._dataspec[name].shape
            return

        size = self._datalog[name].copy()  # to be updated below based on index
        for i, item in enumerate(index):
            if isinstance(item, slice):
                # stop = None means we have written data to this dimension completely
                if item.stop is None:
                    size[i] = self._dataspec[name].shape[i]  # maximum possible value
                else:  # compare with existing size along ith dimension
                    size[i] = max(size[i], item.stop)
            elif item is ...:
                size[i] = self._dataspec[name].shape[i]  # maximum possible value
            else:  # item is an int
                size[i] = max(size[i], item)
        self._datalog[name] = size
        logger.debug(f"Tracked dataset '{name}' size: {self._datalog[name]}.")

    def save_metadata(self, metadataspec: dict[Union[str, None], dict]) -> None:
        """metadataspec is a dict of dicts. outermost dict key = group in data file to save the metadata to. if key is a str, we create a group with that name. if key is None, we save to top level group in the file. value = metadata dict with key-value pair stored as attributes of the group named by their key. every key in the metadata dict must be a string. the following metadata dict values types/structures are recognized and saved according to the convention below (others are ignored and a warning is thrown):
        1. a single Number (int, float, complex) or string or bool is saved as is
        2. a numpy array, within hdf5 size limitations (<64kB) is saved as is
        3. collections (list, tuple, set, frozenset) are all cast to list, and if they contain all types in point (1), we save them as is. else, we convert the list to a dict with the value's index as the key and save this dictionary.
        4. None type is saved as an empty string as hdf5 doesn't have a native None type
        4. a dictionary whose key-value pairs comply with the convention above"""
        self._validate_session()
        file = self._file
        for name, metadata in metadataspec.items():
            group = file if name is None else file.create_group(name, track_order=True)
            self._save_metadata(group, **metadata)

    def _save_metadata(self, group: h5py.Group, **metadata) -> None:
        """internal method, made for recursive saving of metadata"""
        try:
            for key, value in metadata.items():
                value = self._parse_attribute(key, value)
                if isinstance(value, dict):
                    subgroup = group.create_group(key, track_order=True)
                    self._save_metadata(subgroup, **value)
                else:
                    group.attrs[key] = value
                    logger.debug(f"Set {group = } attr '{key}' with {type(value) = }.")
        except ValueError:
            message = (
                f"Got ValueError while saving metadata with {key = } and {value = }. "
                f"Data size is too large (>64kB), please save it as a dataset instead."
            )
            logger.error(message)
            raise DataSavingError(message)
        except TypeError:
            message = (
                f"Got TypeError while saving metadata with {key = } and {value = }. "
                f"This is because h5py does not support this data type / structure."
            )
            logger.error(message)
            raise DataSavingError(message)

    def _parse_attribute(self, key, value):
        """ """
        if isinstance(value, (Number, np.number, str, bool, np.ndarray, dict)):
            return value
        elif isinstance(value, (list, tuple, set, frozenset)):
            value = list(value)
            # if list contains all numbers or strings or bool, return as is
            if all(isinstance(item, (Number, str, bool)) for item in value):
                return value
            else:  # else convert it to a dictionary with the index as the key
                return {str(idx): item for idx, item in enumerate(value)}
        elif value is None:
            return h5py.Empty("S10")
        else:
            logger.warning(
                f"Found unusual {value = } of {type(value)} while parsing metadata "
                f"{key = }, h5py attribute saving behaviour may not be reliable."
            )
            return value
