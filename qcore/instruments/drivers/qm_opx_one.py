from qcore.instruments.instrument import Instrument
from qcore.variables.parameter import Parameter


class OPXOne(Instrument):
    """Dummy instrument containing relevant information for connecting to an OPX+"""


    def __init__(self, name: str, id: str, **parameters):
        super().__init__(id,name=name, **parameters)

    def connect(self) -> None:
        pass
    
    @property
    def status(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass
