"""
This module contains utilities for staging Resources prior to running experiments.
Only .yml config files containing Resource objects can be staged.
The Stage can link up to the Server to retrieve remotely served Resources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import Pyro5.api as pyro

from qcore.helpers.logger import logger
from qcore.helpers import server
import qcore.helpers.yamlizer as yml
import yaml
from qcore.resource import Resource
from qcore.pulses.clear_readout_pulse import ClearReadoutPulse, DoubleClearReadoutPulse
from qcore.pulses.readout_pulse import DoubleConstantReadoutPulse

# these imports are needed for yamlizing to work
from qcore.modes import *
from qcore.instruments import *
from qcore.pulses import *


class StageError(Exception):
    """ """


class Stage:
    """ """

    def __init__(self, configpath: Path = None, remote: bool = False) -> None:
        """
        configpath (Path): .yml config filepath containing Resource objects to be staged
        remote (bool): stage remote Resources by linking up to the remote Server if True
        """
        self._configpath = (
            Path(configpath) if isinstance(configpath, str) else configpath
        )

        # _resources is a dict with key: resource name and
        # value: Resource object for local and Resource proxy for remote resources
        self._resources: dict[str, Union[Resource, pyro.Proxy]] = {}

        # ensure that configpath exists
        if self._configpath is not None:
            self._configpath.parent.mkdir(exist_ok=True)
            self._configpath.touch(exist_ok=True)
            try:
                resources = yml.load(self._configpath)
            except yaml.constructor.ConstructorError as E:
                yml.register(ClearReadoutPulse)
                yml.register(DoubleClearReadoutPulse)
                yml.register(DoubleConstantReadoutPulse)
            if resources:
                self.add(*resources)

        self._server, self._proxies = None, []  # will be updated by _link()
        if remote:
            self._link()

        logger.debug(f"Set up a Stage with {remote = }")

    def _link(self) -> None:
        """ """
        self._server, self._proxies = server.link()
        self.add(*self._proxies)

    def __enter__(self) -> Stage:
        """ """
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """ """
        self.teardown()

    def teardown(self) -> None:
        """ """
        self.save()
        if self._server is not None:
            server.unlink(self._server, *self._proxies)
            logger.debug("Unlinked from the remote stage.")
        logger.debug("Tore down the Stage gracefully!")

    def save(self) -> None:
        """ """
        if self._configpath is not None:
            logger.debug(f"Saving staged resources to {self._configpath}...")
            resources = [r for r in self._resources.values() if isinstance(r, Resource)]
            if resources:
                yml.dump(self._configpath, *resources)

    @property
    def resources(self) -> set[str]:
        """ """
        return set(self._resources.keys())

    def add(self, *resources: Resource) -> None:
        """ """
        for resource in resources:
            name = resource.name
            if name in self._resources:
                message = (
                    f"Unable to stage Resource '{resource}' with {name = } as "
                    f"another resource with the same name exists on this stage."
                )
                logger.error(message)
                raise StageError(message)
            else:
                self._resources[name] = resource
                logger.info(f"Staged resource with '{name = }'.")

    def remove(self, *names: str) -> None:
        """ """
        for name in names:
            if name in self._resources:
                resource = self._resources[name]
                if resource in self._proxies:  # resource is a Proxy object
                    server.release(resource)
                del self._resources[name]
                logger.info(f"Unstaged {resource = }.")
            else:
                message = f"Resource with '{name = }' does not exist on Stage."
                logger.error(message)
                raise StageError(message)

    def get(self, *names: str) -> list[Resource]:
        """ """
        resources = []
        for name in names:
            if name in self._resources:
                resources.append(self._resources[name])
            else:
                message = f"Resource with {name = } does not exist on Stage."
                logger.error(message)
                raise StageError(message)
        return resources
