""" Instrument server """

from pathlib import Path

import Pyro5.api as pyro
import Pyro5.errors as pyro_errors

from qcore.helpers.logger import logger
import qcore.helpers.yamlizer as yml
from qcore.instruments.instrument import Instrument
from qcore.resource import Resource
from qcore.variables.parameter import Parameter


@pyro.expose
class Server:
    """ """

    NAME = "SERVER"
    PORT = 9090  # port to bind a remote server on, used to initialize Pyro Daemon
    URI = f"PYRO:{NAME}@localhost:{PORT}"  # unique resource identifier (URI)

    def __init__(self, configpath: Path) -> None:
        """ """
        self._instruments: list[Instrument] = yml.load(configpath)
        self._daemon = pyro.Daemon(port=Server.PORT)
        self._services: list[pyro.URI] = []  # list of instrument URIs, set by _serve()

    def serve(self) -> None:
        """blocking function"""
        self._expose()
        for instrument in self._instruments:
            uri = self._daemon.register(instrument, objectId=instrument.name)
            self._services.append(uri)
            logger.info(f"Registered {instrument = } with daemon at {uri = }.")
        self._daemon.register(self, objectId=Server.NAME)
        with self._daemon:
            logger.info("Remote server setup complete! Now listening for requests...")
            self._daemon.requestLoop()

    def _expose(self) -> None:
        """ """
        classes = {instrument.__class__ for instrument in self._instruments}
        classes |= {Instrument, Resource, Parameter}
        for cls in classes:
            pyro.expose(cls)
            logger.info(f"Exposed class {cls} to Pyro5.")

    @property
    def services(self) -> list[pyro.URI]:
        """ """
        return self._services.copy()

    def teardown(self) -> None:
        """ """
        logger.info("Tearing down the remote server...")
        self._disconnect()
        with self._daemon:
            self._daemon.shutdown()
        logger.info("Remote server teardown complete!")

    def _disconnect(self) -> None:
        """ """
        logger.info("Disonnecting instruments...")
        for instrument in self._instruments:
            print(instrument)
            if instrument.status:
                instrument.disconnect()


def link() -> tuple[pyro.Proxy, list[pyro.Proxy]]:
    """ """
    server = pyro.Proxy(Server.URI)
    try:
        services = server.services
    except pyro_errors.CommunicationError as err:  # no remote server found
        logger.error(f"Remote server requested but not found at {Server.URI}")
        raise err from None
    else:
        instruments = [pyro.Proxy(uri) for uri in services]
        return (server, instruments)


def unlink(server: pyro.Proxy, *instruments: pyro.Proxy) -> None:
    """ """
    server._pyroRelease()
    for instrument in instruments:
        instrument._pyroRelease()
