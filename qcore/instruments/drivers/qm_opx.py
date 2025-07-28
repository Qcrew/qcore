from typing import Literal, Union

from qm import DictQuaConfig
from qcore.instruments.instrument import Instrument
from qcore.variables.parameter import Parameter


class OPXPlus(Instrument):
    """Dummy instrument containing relevant information for connecting to an OPX+"""
    cluster_name: str = Parameter()
    type: str = Parameter()

    def __init__(self, cluster_name: str, id: str, **parameters):
        self._cluster_name = cluster_name
        self._type = 'opx_plus'
        super().__init__(id, **parameters)

    @cluster_name.getter
    def cluster_name(self) -> str:
        """ """
        return self._cluster_name

    @cluster_name.setter
    def cluster_name(self, value: str) -> None:
        """ """
        self._cluster_name = value

    @type.getter
    def type(self) -> str:
        return self._type

    @property
    def status(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


class OPX1000(Instrument):
    """Dummy instrument containing relevant information for connecting to an OPX1000"""
    settings: DictQuaConfig = Parameter()
    cluster_name: str = Parameter()
    type: str = Parameter()

    def __init__(self, cluster_name: str, id: str, settings: DictQuaConfig = None, **parameters):
        """
        Args:
            cluster_name (str):
                The name of the OPX1000 cluster
            id (str):
                The IP address of the OPX1000 cluster
            settings (DictQuaConfig, optional):
                The `settings` attribute is a partial QUA config dictionary which will
                be deep-merged into the one built from the modes. It can be used to
                override individual OPX1000 config parameters, or set those which are
                not accessible through the `Mode`s.
        """
        if settings is None:
            settings = {}
        self._settings = settings
        self._cluster_name = cluster_name
        self._type = 'opx1000'
        super().__init__(id, **parameters)

    @settings.getter
    def settings(self) -> DictQuaConfig:
        """ """
        return self._settings
    
    @settings.setter
    def settings(self, settings:dict) -> None:
        """ """
        self._settings = settings

    @cluster_name.getter
    def cluster_name(self) -> str:
        """ """
        return self._cluster_name

    @cluster_name.setter
    def cluster_name(self, value: str) -> None:
        """ """
        self._cluster_name = value

    @type.getter
    def type(self) -> str:
        return self._type

    @property
    def status(self) -> bool:
        return True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

OPX = Union[OPXPlus, OPX1000]
